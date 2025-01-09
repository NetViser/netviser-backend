import uuid
from app.services.redis_service import RedisClient
from fastapi import APIRouter, File, Response, UploadFile, Cookie, Body
from typing import Dict, Any, Optional
from app.configs.config import get_settings
import xgboost as xgb
import pandas as pd
import os

router = APIRouter(prefix="/api", tags=["api"])
settings = get_settings()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    converted_file = pd.read_csv(file.file, engine="pyarrow", dtype_backend="pyarrow")

    X_test_scaled_array = converted_file.values
    dmatrix = xgb.DMatrix(X_test_scaled_array)

    model = xgb.Booster()
    load_model = os.path.join("model", "xgb_booster.model")
    model.load_model(load_model)
    predictions = model.predict(dmatrix)
    print(predictions)
    return {"file_name": file.filename}

# Since we don't have real login, we'll store a custom "session_id" in a cookie
SESSION_COOKIE_NAME = "session_id"

# Instantiate our singleton Redis client once
redis_client = RedisClient()

@router.post("/store-json")
async def store_json(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    payload: Dict[str, Any] = Body(...)
):
    """
    Stores the provided JSON payload in a Redis session. 
    If there is no existing session_id, a new one is created.
    """
    if not session_id:
        # If the user doesn't have a session_id, create a new one
        session_id = str(uuid.uuid4())
        # Set the cookie with a 5-minute expiration
        response.set_cookie(key=SESSION_COOKIE_NAME, value=session_id, max_age=300)
    
    # Store or update the session data in Redis with a 5-minute TTL
    redis_client.set_session_data(session_id, payload, ttl_in_seconds=300)

    return {
        "message": "JSON payload successfully stored in session.",
        "session_id": session_id
    }

@router.get("/current-session")
async def get_session(
    session_id: str = Cookie(None)
):
    if not session_id:
        return {"message": "No session found"}
    data = redis_client.get_session_data(session_id)
    if not data:
        return {"message": "Session expired or not found"}
    return {"session_data": data}


@router.post("/end-session")
async def end_session(
    session_id: str = Cookie(None),
    response: Response = None
):
    if session_id:
        redis_client.delete_session_data(session_id)
        # Optionally clear the cookie
        response.delete_cookie(SESSION_COOKIE_NAME)
    return {"message": "Session ended"}