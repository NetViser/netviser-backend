from pydantic import BaseModel
from typing import List

class SampleFileResponse(BaseModel):
    name: str
    featuredAttacks: List[str]

    class Config:
        # Enable compatibility with ORM-like objects if needed
        from_attributes = True

class SampleFilesListResponse(BaseModel):
    sample_files: List[SampleFileResponse]

    class Config:
        from_attributes = True