from database import Base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.sql import func

class Users(Base):
    __tablename__ = 'tu_users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(50), unique=True)
    username = Column(String(50), unique=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    hashed_password = Column(String(250))
    
class Post(Base):
    __tablename__ = 'tu_posts'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(50))
    content = Column(String(100))
    user_id = Column(Integer, ForeignKey("tu_users.id"))
    
class Chatbot(Base):
    __tablename__ = 'tu_chatbot'

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50))
    content = Column(String(450))
    seed = Column(String(50))
    timestamp = Column(DateTime, default=func.now())  # Populates with current timestamp
        