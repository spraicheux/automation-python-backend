from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(request: LoginRequest):
    if request.email == "admin@gmail.com" and request.password == "admin@samuel":
        return {"token": "valid-token"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
