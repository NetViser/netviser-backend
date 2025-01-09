from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    AWS_REGION: Optional[str] = None

    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    REDIS_USERNAME: Optional[str] = None
    REDIS_PASSWORD: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    return Settings()
