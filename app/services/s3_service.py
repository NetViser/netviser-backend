from io import BytesIO
from typing import Any, Dict, Optional
from botocore.config import Config
import uuid
import boto3
from botocore.exceptions import ClientError
from fastapi import Cookie, HTTPException, UploadFile
import asyncio
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor

from app.configs.config import get_settings


settings = get_settings()

# Initialize a custom ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=50, thread_name_prefix="s3-upload-")


class S3:
    """
    Singleton class for interacting with AWS S3.
    """

    _instance = None

    def __new__(cls):
        """Create a new instance of the S3 Service class if it doesn't already exist."""
        if cls._instance is None:
            cls._instance = super(S3, cls).__new__(cls)
            cls._instance.initialize_client()
        return cls._instance

    def initialize_client(self):
        """Initialize the boto3 client for S3."""
        bucket_name = settings.S3_BUCKET_NAME
        access_key = settings.AWS_ACCESS_KEY_ID
        secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        region_name = settings.AWS_REGION

        if not all([access_key, secret_access_key, region_name]):
            raise ValueError("Some environment variables are missing.")

        if bucket_name is None:
            raise ValueError("bucket_name is required")

        self.bucket_name = bucket_name
        self.region_name = region_name
        self.path_prefix = "uploads"

        # Create a botocore Config with Signature Version 4
        my_config = Config(signature_version="s3v4")

        self.client = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
            config=my_config,
            endpoint_url=f"https://s3.{region_name}.amazonaws.com",
        )

        # Configure TransferConfig for multipart uploads
        self.transfer_config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # 100 MB
            multipart_chunksize=50 * 1024 * 1024,  # 50 MB
            max_concurrency=10,
            use_threads=True,
            # Adjust as needed
        )

    async def read(self, s3_key: str) -> bytes:
        """Download a file from S3."""
        buffer = BytesIO()
        try:
            await asyncio.to_thread(
                self.client.download_fileobj, self.bucket_name, s3_key, buffer
            )
            buffer.seek(0)
            return buffer.read()
        except ClientError as e:
            raise HTTPException(status_code=404, detail="File not found in S3.") from e

    async def generate_presigned_url(
        self,
        filename: str,
        expiration: int = 3600,
        session_id: Optional[str] = Cookie(None),
    ) -> str:
        """Generate a presigned URL for accessing a file in S3."""

        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID missing")

        file_id = str(uuid.uuid4())
        s3_key = f"sessions/{session_id}/{file_id}_{filename}"

        try:
            presigned_url = self.client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return presigned_url
        except ClientError as e:
            raise HTTPException(
                status_code=500, detail="Failed to generate presigned URL."
            ) from e

    def get_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        session_id: Optional[str] = Cookie(None),
    ) -> str:
        try:
            print("Get url", f"{self.path_prefix}/{session_id}/{s3_key}")
            url = self.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": f"{self.path_prefix}/{session_id}/{s3_key}",
                },
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            raise HTTPException(
                status_code=500, detail="Failed to generate get URL."
            ) from e

    async def upload(
        self,
        file: UploadFile,
        file_path: str,
        session_id: Optional[str] = Cookie(None),
        extra_args: dict = {},
    ) -> Dict[str, Any]:
        """
        Upload a file to S3 with optional extra arguments like ACL or ContentType.
        Args:
            file (UploadFile): File to be uploaded.
            filename (str): Key to save the file as in S3.
            extra_args (dict, optional): Additional S3 arguments like ACL or ContentType.
        """
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID missing")

        try:
            # Generate a unique S3 key
            s3_key = f"{self.path_prefix}/{session_id}/{file_path}"

            # Upload the file to S3 using multipart upload
            await asyncio.to_thread(
                self.client.upload_fileobj,
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args,
                Config=self.transfer_config,
            )

            return {
                "filename": file_path,
                "s3_key": s3_key,
                "file_path": f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}",
                "extra_args": extra_args,
            }
        except ClientError as e:
            raise HTTPException(
                status_code=500, detail="Failed to upload file to S3."
            ) from e

    async def file_exists(self, filename: str, session_id) -> bool:
        """Check if a file exists in the S3 bucket."""
        try:
            response = await asyncio.to_thread(
                self.client.list_objects_v2,
                Bucket=self.bucket_name,
                Prefix=f"{self.path_prefix}/{session_id}/{filename}",
            )
            return "Contents" in response
        except ClientError as e:
            raise HTTPException(
                status_code=500, detail="Error checking file existence in S3."
            ) from e

    async def get_unique_filename(self, filename: str) -> str:
        """Generate a unique filename if a file with the same name exists."""
        base_name, extension = (
            filename.rsplit(".", 1) if "." in filename else (filename, "")
        )
        counter = 1

        new_filename = filename
        while await self.file_exists(new_filename):
            new_filename = f"{base_name}({counter})"
            if extension:
                new_filename += f".{extension}"
            counter += 1

        return new_filename

    async def delete(self, filename: str):
        """Delete a file from S3."""
        try:
            await asyncio.to_thread(
                self.client.delete_object, Bucket=self.bucket_name, Key=filename
            )
        except ClientError as e:
            raise HTTPException(
                status_code=500, detail="Failed to delete file from S3."
            ) from e
