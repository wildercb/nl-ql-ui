from pydantic import BaseModel
from typing import Optional, List

class TranslationResult(BaseModel):
    original_query: str
    graphql_query: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    explanation: Optional[str] = None
    warnings: List[str] = []
    suggested_improvements: List[str] = []

    class Config:
        arbitrary_types_allowed = True 