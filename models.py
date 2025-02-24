from pydantic import BaseModel

class ImageRequest(BaseModel):
    data: str
