from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import jwt
import bcrypt

from database import get_db
from models import User, Message, Conversation, ConversationMessage

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

SECRET_KEY = "your-secret-key-lex-assist"  # In production, use env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

# --- Schemas ---
class UserCreate(BaseModel):
    email: str
    password: str
    role: str
    name: str
    specialization: Optional[str] = None
    experience_years: Optional[int] = None
    practicing_courts: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    role: str
    name: str
    email: str

class LawyerResponse(BaseModel):
    id: int
    name: str
    email: str
    specialization: Optional[str] = None
    experience_years: Optional[int] = None
    practicing_courts: Optional[str] = None

class MessageCreate(BaseModel):
    receiver_id: int
    content: str

class MessageResponse(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    content: str
    timestamp: datetime

# --- Utils ---
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# --- Endpoints ---
@router.post("/signup", response_model=Token)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    if user.role not in ["customer", "lawyer"]:
        raise HTTPException(status_code=400, detail="Invalid role")
        
    hashed_password = get_password_hash(user.password)
    new_user = User(
        email=user.email, 
        password_hash=hashed_password, 
        role=user.role, 
        name=user.name,
        specialization=user.specialization if user.role == "lawyer" else None,
        experience_years=user.experience_years if user.role == "lawyer" else None,
        practicing_courts=user.practicing_courts if user.role == "lawyer" else None
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.email, "role": new_user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user_id": new_user.id, "role": new_user.role, "name": new_user.name, "email": new_user.email}


@router.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email, "role": db_user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user_id": db_user.id, "role": db_user.role, "name": db_user.name, "email": db_user.email}

@router.get("/lawyers", response_model=List[LawyerResponse])
def get_lawyers(db: Session = Depends(get_db)):
    lawyers = db.query(User).filter(User.role == "lawyer").all()
    return lawyers

@router.post("/hire/{lawyer_id}")
def hire_lawyer(lawyer_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    lawyer = db.query(User).filter(User.id == lawyer_id, User.role == "lawyer").first()
    if not lawyer:
        raise HTTPException(status_code=404, detail="Lawyer not found")
    
    initial_msg = Message(sender_id=current_user.id, receiver_id=lawyer_id, content=f"Hi {lawyer.name}, I would like to hire your legal services.")
    db.add(initial_msg)
    db.commit()
    return {"status": "success", "message": f"Successfully sent hiring request to {lawyer.name}"}

@router.post("/messages", response_model=MessageResponse)
def send_message(msg: MessageCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    receiver = db.query(User).filter(User.id == msg.receiver_id).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    new_msg = Message(sender_id=current_user.id, receiver_id=msg.receiver_id, content=msg.content)
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    return new_msg

@router.get("/messages/contacts")
def get_contacts(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sent_msgs = db.query(Message.receiver_id).filter(Message.sender_id == current_user.id).distinct().all()
    recv_msgs = db.query(Message.sender_id).filter(Message.receiver_id == current_user.id).distinct().all()
    
    contact_ids = set([r[0] for r in sent_msgs] + [r[0] for r in recv_msgs])
    if not contact_ids:
        return []
    
    contacts = db.query(User).filter(User.id.in_(contact_ids)).all()
    return [{"id": c.id, "name": c.name, "role": c.role} for c in contacts]

@router.get("/messages/{other_user_id}", response_model=List[MessageResponse])
def get_conversation(other_user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    messages = db.query(Message).filter(
        or_(
            and_(Message.sender_id == current_user.id, Message.receiver_id == other_user_id),
            and_(Message.sender_id == other_user_id, Message.receiver_id == current_user.id)
        )
    ).order_by(Message.timestamp.asc()).all()
    return messages

# --- Conversation (RAG Chat) Endpoints ---

class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    expires_at: Optional[datetime]

class ConversationDetail(ConversationResponse):
    messages: List[dict]

@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Also clean up expired conversations
    now = datetime.utcnow()
    db.query(Conversation).filter(Conversation.expires_at < now).delete()
    db.commit()
    
    return db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.created_at.desc()).all()

@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    expires_at = datetime.utcnow() + timedelta(days=current_user.chat_retention_days)
    new_conv = Conversation(user_id=current_user.id, expires_at=expires_at)
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return new_conv

@router.get("/conversations/{conv_id}", response_model=ConversationDetail)
def get_conversation_history(conv_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    msgs = db.query(ConversationMessage).filter(ConversationMessage.conversation_id == conv_id).order_by(ConversationMessage.timestamp.asc()).all()
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "expires_at": conv.expires_at,
        "messages": [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in msgs]
    }

@router.post("/settings/retention")
def update_retention(days: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if days < 1 or days > 7:
        raise HTTPException(status_code=400, detail="Retention must be between 1 and 7 days")
    current_user.chat_retention_days = days
    db.commit()
    return {"status": "success", "days": days}
