from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from typing import Optional
from cryptography.fernet import Fernet
from schemas.schema import UserBase, PostBase, BotBase
import random
from utils import generate_llmanswer, generate_llmanswer_sin_URLSource

app = FastAPI()
key = Fernet.generate_key()
f = Fernet(key)

models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@app.post("/users/", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserBase, db:db_dependency):
    user_data = user.dict() 
    new_user = {"username": user_data["username"], "email": user_data["email"], "first_name": user_data["first_name"], "last_name": user_data["last_name"]}
    new_user["hashed_password"] = f.encrypt(user_data["hashed_password"].encode("utf-8"))
    #print(new_user)
    db_user = models.Users(**new_user)
    db.add(db_user)
    db.commit()

@app.post("/chatbot/", status_code=status.HTTP_201_CREATED)
async def create_message(bot_obj: BotBase, db:db_dependency):
    bot_data = bot_obj.dict() 
    new_bot_user = {"content": bot_data["content"]}
    new_bot_user["role"] = "user"
    new_bot_user["seed"] = "13" #random.randint(10000, 99999)  # Inclusive range
    #print(new_bot_user)
    #print(type(new_bot_user))
    #--insert into DB user question
    db_bot_usermsg = models.Chatbot(**new_bot_user)
    db.add(db_bot_usermsg)
    db.commit()
    #send params to util function to elaborate a LLM answer
    user_question = await generate_llmanswer(f"role: {new_bot_user['role']}, content: {new_bot_user['content']}",seed_number=new_bot_user['seed'],user_query=new_bot_user['content'])
    #insert into DB system question
    new_bot_system = {"role": "system","content": user_question,"seed": new_bot_user['seed']}
    #print(type(new_bot_system))
    #print(new_bot_system)
    db_bot_systemmsg = models.Chatbot(**new_bot_system)
    db.add(db_bot_systemmsg)
    db.commit()
    return {"message": "ok"}
    
@app.get("/chatbot/{seed_id}", status_code=status.HTTP_200_OK)
async def get_seed_hist(seed_id: int, db:db_dependency):
    hist_seed = db.query(models.Chatbot).filter(models.Chatbot.seed == seed_id).all()
    if hist_seed is None:
        raise HTTPException(status_code=404, detail='Seed not found')   
    return hist_seed  