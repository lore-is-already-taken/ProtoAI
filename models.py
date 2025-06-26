from pydantic import BaseModel
from typing import Optional

class ImageRequest(BaseModel):
    data: str
    model: Optional[str] = None