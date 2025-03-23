from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    REDIS_USERNAME: Optional[str] = None
    REDIS_PASSWORD: Optional[str] = None

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: Optional[str] = None

    ENV: Optional[str] = None

    # Google Cloud Storage
    GCS_BUCKET_NAME: Optional[str] = None
    GCS_CREDENTIALS_JSON_PATH: Optional[str] = None
    GC_PROJECT_ID: Optional[str] = None
    
    # New properties in uppercase with default values
    SECURE_COOKIE: bool = False
    SAMESITE: str = 'lax'

    model_config = SettingsConfigDict(env_file=".env")

    def __init__(self, **values):
        super().__init__(**values)
        # Set SECURE_COOKIE and SAMESITE based on ENV
        if self.ENV == "production":
            self.SECURE_COOKIE = True
            self.SAMESITE = "none"
        elif self.ENV == "local":
            self.SECURE_COOKIE = False
            self.SAMESITE = "lax"


def get_settings() -> Settings:
    return Settings()