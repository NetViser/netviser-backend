from io import BytesIO
from typing import Any, Dict, Optional
import uuid
import boto3
from botocore.exceptions import ClientError
from fastapi import Cookie, HTTPException, UploadFile
import asyncio

from app.configs.config import get_settings


settings = get_settings()


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

        self.client = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )

    async def read(self, filename: str) -> bytes:
        """Download a file from S3."""
        buffer = BytesIO()
        try:
            await asyncio.to_thread(
                self.client.download_fileobj, self.bucket_name, filename, buffer
            )
            buffer.seek(0)
            return buffer.read()
        except ClientError as e:
            raise e

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
        s3_key = f"uploads/{session_id}/{file_id}_{filename}"

        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )
        except ClientError as e:
            raise e

    async def upload(
        self,
        file: UploadFile,
        filename: str,
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
        try:
            # Check if the file already exists in S3
            if await self.file_exists(filename):
                filename = await self.get_unique_filename(filename)

            file_id = str(uuid.uuid4())
            s3_key = f"uploads/{session_id}/{file_id}_{filename}"

            await asyncio.to_thread(
                self.client.upload_fileobj,
                file.file,
                self.bucket_name,
                filename,
                ExtraArgs=extra_args,
            )

            return {
                "filename": filename,
                "file_path": f"https://{self.bucket_name}-buckets.s3.{self.region_name}.amazonaws.com/"
                + s3_key,
                "extra_args": extra_args,
            }
        except ClientError as e:
            raise e

    async def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the S3 bucket."""
        try:
            response = await asyncio.to_thread(
                self.client.list_objects_v2,
                Bucket=self.bucket_name,
                Prefix=filename,
            )
            return "Contents" in response
        except ClientError as e:
            raise e

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
            raise e