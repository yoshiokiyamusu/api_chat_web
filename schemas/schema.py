from typing import Optional
from pydantic import BaseModel



class PostBase(BaseModel):
    title: str
    content: str
    user_id: int

class UserBase(BaseModel):
    id: Optional[int] = None
    username: str
    email: str
    first_name: str
    last_name: str
    hashed_password: str    

class BotBase(BaseModel):
    id: Optional[int] = None
    role: Optional[str] = None
    content: str
    seed: Optional[str] = None
        