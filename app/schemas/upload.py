from pydantic import BaseModel

class UploadSampleResponse(BaseModel):
    completed: bool
    message: str
    session_id: str
    bucket_key: str

class UploadPresignedResponse(BaseModel):
    completed: bool
    message: str
    session_id: str
    presigned_url: str
    bucket_key: str

class UploadCompleteResponse(BaseModel):
    message: str  # "File processed successfully"
    session_id: str  # UUID of the session
    bucket_key: str  # GCS key for the processed (model-applied) file