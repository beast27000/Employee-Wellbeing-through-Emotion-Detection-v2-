from pydantic import BaseModel

class LoginData(BaseModel):
    employee_id: str
    department: str

class FeedbackLoginData(BaseModel):
    emp_id: str
    department: str