#Previous :

# from pydantic import BaseModel
# from datetime import datetime
# from typing import Optional

# class PredictionResultBase(BaseModel):
#     filename: Optional[str] = None
#     predicted_class: str
#     confidence: float

# class PredictionResultCreate(PredictionResultBase):
#     pass

# class PredictionResultResponse(PredictionResultBase):
#     id: int
#     timestamp: datetime

#     class Config:
#         from_attributes = True

#Updated :

from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserCreateResponse(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime
    model_config = ConfigDict(from_attributes= True)
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id: Optional[int] = None

class PredictionResultBase(BaseModel):
    filename: Optional[str] = None
    predicted_class: str
    confidence: float

class PredictionResultResponse(PredictionResultBase):
    id: int
    timestamp: datetime
    user_id: int
    owner: UserCreateResponse 

    model_config = ConfigDict(from_attributes=True)