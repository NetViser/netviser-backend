from io import BytesIO
import os
from typing import Any, Dict, Optional
from fastapi import Cookie, HTTPException, UploadFile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google.oauth2 import service_account
from google.cloud import storage
from google.auth import default
from google.auth.transport import requests

import logging

# Setup basic logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.configs.config import get_settings

settings = get_settings()

# Initialize a custom ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=50, thread_name_prefix="gcs-upload-")

class GCS:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GCS, cls).__new__(cls)
            cls._instance.initialize_client()
        return cls._instance

    def initialize_client(self):
        bucket_name = settings.GCS_BUCKET_NAME
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME is missing.")

        self.bucket_name = bucket_name
        self.path_prefix = "uploads"

        if settings.ENV == "local" and hasattr(settings, "GCS_CREDENTIALS_JSON_PATH"):
            credentials_path = settings.GCS_CREDENTIALS_JSON_PATH
            if credentials_path and os.path.exists(credentials_path):
                logger.info(f"Local env detected. Using service account credentials from file: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.client = storage.Client(credentials=credentials, project=settings.GC_PROJECT_ID)
                self.use_adc = False
            else:
                raise ValueError(f"Local env detected but credentials file not found at: {credentials_path}")
        else:
            logger.info("Non-local env detected. Using ADC with token signing workaround")
            credentials, project_id = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if credentials.token is None:
                logger.info("Refreshing credentials to obtain access token")
                auth_request = requests.Request()
                credentials.refresh(auth_request)

            self.credentials = credentials
            self.service_account_email = getattr(credentials, "service_account_email", None)
            if not self.service_account_email:
                raise ValueError("Service account email not found in credentials")
            self.client = storage.Client(credentials=credentials, project=settings.GC_PROJECT_ID)
            self.use_adc = True

        self.bucket = self.client.bucket(bucket_name)
        logger.info(f"GCS Client initialized for bucket: {bucket_name}")

    async def generate_presigned_url(self, file_path: str, expiration: int = 3600, session_id: Optional[str] = Cookie(None)) -> str:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID missing")

        gcs_key = f"{self.path_prefix}/{session_id}/{file_path}"
        MAX_FILE_SIZE = 1_073_741_824  # 1GB in bytes

        try:
            logger.info(f"Generating signed PUT URL for: {gcs_key} with max size 1GB")
            blob = self.bucket.blob(gcs_key)
            conditions = [["content-length-range", 0, MAX_FILE_SIZE]]
            if self.use_adc:
                url = blob.generate_signed_url(
                    expiration=expiration,
                    method="PUT",
                    version="v4",
                    service_account_email=self.service_account_email,
                    access_token=self.credentials.token,
                    conditions=conditions
                )
            else:
                url = blob.generate_signed_url(
                    expiration=expiration,
                    method="PUT",
                    version="v4",
                    conditions=conditions
                )
            logger.info(f"Generated presigned URL: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed PUT URL: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL.") from e

    async def read(self, gcs_key: str) -> bytes:
        buffer = BytesIO()
        try:
            logger.info(f"Reading file from GCS: {gcs_key}")
            blob = self.bucket.blob(gcs_key)
            await asyncio.to_thread(blob.download_to_file, buffer)
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"Error reading file from GCS: {gcs_key}, Error: {e}")
            raise HTTPException(status_code=404, detail="File not found in GCS.") from e

    def get_url(self, gcs_key: str, expiration: int = 3600, session_id: Optional[str] = Cookie(None)) -> str:
        try:
            full_key = f"{self.path_prefix}/{session_id}/{gcs_key}"
            logger.info(f"Generating signed GET URL for: {full_key}")
            blob = self.bucket.blob(full_key)
            if self.use_adc:
                url = blob.generate_signed_url(
                    expiration=expiration,
                    method="GET",
                    version="v4",
                    service_account_email=self.service_account_email,
                    access_token=self.credentials.token
                )
            else:
                url = blob.generate_signed_url(expiration=expiration, method="GET", version="v4")
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed GET URL: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate get URL.") from e

    async def upload(self, file: UploadFile, file_path: str, session_id: Optional[str] = Cookie(None), extra_args: dict = {}) -> Dict[str, Any]:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID missing")

        try:
            gcs_key = f"{self.path_prefix}/{session_id}/{file_path}"
            logger.info(f"Uploading file to GCS: {gcs_key}")
            blob = self.bucket.blob(gcs_key)

            if "ContentType" in extra_args:
                blob.content_type = extra_args["ContentType"]

            file_content = await file.read()
            buffer = BytesIO(file_content)

            await asyncio.to_thread(blob.upload_from_file, buffer)

            logger.info(f"Upload successful: {gcs_key}")
            return {
                "filename": file_path,
                "gcs_key": gcs_key,
                "file_path": f"https://storage.googleapis.com/{self.bucket_name}/{gcs_key}",
                "extra_args": extra_args,
            }
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {gcs_key}, Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload file to GCS.") from e

    async def file_exists(self, filename: str, session_id) -> bool:
        try:
            prefix = f"{self.path_prefix}/{session_id}/{filename}"
            logger.info(f"Checking file existence: {prefix}")
            blobs = await asyncio.to_thread(list, self.client.list_blobs(self.bucket_name, prefix=prefix))
            exists = len(blobs) > 0
            logger.info(f"File exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking file existence in GCS: {e}")
            raise HTTPException(status_code=500, detail="Error checking file existence in GCS.") from e

    async def object_exists(self, gcs_key: str) -> bool:
        try:
            logger.info(f"Checking if object exists: {gcs_key}")
            blob = self.bucket.blob(gcs_key)
            exists = await asyncio.to_thread(blob.exists)
            logger.info(f"Object exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking object existence: {e}")
            raise HTTPException(status_code=500, detail="Error checking object existence in GCS.") from e

    async def get_unique_filename(self, filename: str, session_id) -> str:
        base_name, extension = (
            filename.rsplit(".", 1) if "." in filename else (filename, "")
        )
        counter = 1
        new_filename = filename

        while await self.file_exists(new_filename, session_id):
            new_filename = f"{base_name}({counter})"
            if extension:
                new_filename += f".{extension}"
            counter += 1

        logger.info(f"Unique filename resolved: {new_filename}")
        return new_filename

    async def delete(self, filename: str):
        try:
            logger.info(f"Deleting file: {filename}")
            blob = self.bucket.blob(filename)
            await asyncio.to_thread(blob.delete)
            logger.info("File deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting file: {filename}, Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete file from GCS.") from e