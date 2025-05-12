from pydantic import BaseModel
from typing import List, Any, Dict, Optional

class ChatMessage(BaseModel):
    """
    Pydantic model for chat message requests.
    """
    message: str

class ValidationResultItem(BaseModel):
    """
    Pydantic model for a single validation result item.
    """
    field: str
    value: Any
    isValid: Optional[bool]
    reason: str

class ValidationSummaryResponse(BaseModel):
    """
    Pydantic model for the response containing the validation summary.
    """
    validation_summary: str

class GuidelineProcessingResponse(BaseModel):
    """
    Pydantic model for the response after processing guidelines.
    """
    message: str

class ChatResponse(BaseModel):
    """
    Pydantic model for chat message responses.
    """
    response: str

class ErrorDetail(BaseModel):
    """
    Pydantic model for error responses.
    """
    detail: str
