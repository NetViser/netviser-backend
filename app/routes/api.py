import io
import uuid

from fastapi.responses import JSONResponse
import numpy as np
from app.services.redis_service import RedisClient
from fastapi import APIRouter, File, HTTPException, Response, UploadFile, Cookie, Depends
from typing import Optional
from app.configs.config import get_settings
import xgboost as xgb
import pandas as pd
import os
import joblib
from app.services.s3_service import S3

router = APIRouter(prefix="/api", tags=["api"])
settings = get_settings()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    # Read and process the uploaded file
    converted_file = pd.read_csv(file.file, engine="pyarrow", dtype_backend="pyarrow")
    X_test_scaled_array = converted_file.values
    dmatrix = xgb.DMatrix(X_test_scaled_array)

    # Load the pre-trained XGBoost model
    model = xgb.Booster()
    load_model = os.path.join("model", "xgb_booster.model")
    model.load_model(load_model)

    # Make predictions
    predictions = model.predict(dmatrix)
    predictions_y = np.argmax(predictions, axis=1)
    pickle_file_path = "model\label_encoder.pkl"

    encoder = joblib.load(pickle_file_path)

    print(encoder.classes_)

    og_class = encoder.inverse_transform(predictions_y)

    print(og_class)

    converted_file["class"] = og_class

    return JSONResponse(
        content=converted_file.head(1000).to_json(orient="records"),
        media_type="application/json",
    )


# Since we don't have real login, we'll store a custom "session_id" in a cookie
SESSION_COOKIE_NAME = "session_id"

# Instantiate our singleton Redis client once
redis_client = RedisClient()


@router.post("/upload")
async def upload_file(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    file: UploadFile = File(...),
    s3_service: S3 = Depends(S3),
):
    """
    Store file in S3
    """
    if not file:
        raise ValueError("file missing")

    if not session_id:
        # If the user doesn't have a session_id, create a new one
        session_id = str(uuid.uuid4())
        # Set the cookie with a 5-minute expiration
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=300,
            httponly=True,      # Prevents JavaScript access
            secure=(False if settings.STAGE == "local" else True),      # Set to True in production
            samesite="lax",     # Adjust as needed
        )

    try:
        print("Reading and processing the uploaded file...")
        converted_file = pd.read_csv(
            file.file, engine="pyarrow", dtype_backend="pyarrow"
        )
        X_test_scaled_array = converted_file.values
        dmatrix = xgb.DMatrix(X_test_scaled_array)

        # Load the pre-trained XGBoost model
        model = xgb.Booster()
        load_model = os.path.join("model", "xgb_booster.model")
        model.load_model(load_model)

        # Make predictions
        predictions = model.predict(dmatrix)
        predictions_y = np.argmax(predictions, axis=1)
        pickle_file_path = "model\label_encoder.pkl"

        encoder = joblib.load(pickle_file_path)

        og_class = encoder.inverse_transform(predictions_y)

        converted_file["class"] = og_class

        # Convert DataFrame to a CSV in-memory
        buffer = io.BytesIO()
        converted_file.to_csv(buffer, index=False)
        buffer.seek(0)

        # Convert the in-memory CSV back to a file-like object for S3 upload
        csv_file = UploadFile(
            filename=f"{file.filename}_processed.csv",
            file=buffer,
        )

        upload_output = await s3_service.upload(csv_file, csv_file.filename, session_id)
        s3_key = upload_output.get("s3_key")
        print(f"File uploaded to S3 with key: {s3_key}")

        # Store or update the session data in Redis with a 5-minute TTL
        redis_client.set_session_data(session_id, s3_key, ttl_in_seconds=600)
        print("Session data stored in Redis.")
        return {
            "message": "JSON payload successfully stored in session.",
            "session_id": session_id,
        }

    except Exception as e:
        return {
            "message": "Failed to process the uploaded file.",
        }

@router.post("/raw-file-upload")
async def raw_file_upload(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    file: UploadFile = File(...),
    s3_service: S3 = Depends(S3),
):
    """
    Store the raw file in S3
    """
    if not file:
        raise ValueError("file missing")
    

    if not session_id:
        print("No session ID found.")
        # If the user doesn't have a session_id, create a new one
        session_id = str(uuid.uuid4())
        # Set the cookie with a 5-minute expiration
        print("New session ID created:", session_id)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=300,
            httponly=True,      # Prevents JavaScript access
            secure=(False if settings.STAGE == "local" else True),  
            samesite="lax",     # Adjust as needed
        )


    try:
        # Upload the raw file to S3
        upload_output = await s3_service.upload(file, file.filename, session_id)
        s3_key = upload_output.get("s3_key")
        print(f"File uploaded to S3 with key: {s3_key}")

        # Store or update the session data in Redis with a 5-minute TTL
        redis_client.set_session_data(session_id, s3_key, ttl_in_seconds=600)
        print("Session data stored in Redis.")
        return {
            "message": "File successfully stored in session.",
            "session_id": session_id,
        }

    except Exception as e:
        return {
            "message": "Failed to process the uploaded file.",
        }

SESSION_COOKIE_NAME = "session_id"

def get_session_id(session_id: Optional[str] = Cookie(None)) -> str:
    """
    Dependency to extract the session_id from HTTP-only cookies.

    Raises:
        HTTPException: If the session_id cookie is missing.

    Returns:
        str: The extracted session_id.
    """
    if not session_id:
        raise HTTPException(
            status_code=401,
            detail="Session ID missing. Please log in.",
        )
    return session_id

@router.get("/get-file-name")
async def get_file_name(session_id: str = Depends(get_session_id)):
    """
    Retrieve the file name or value pair stored in the session data based on the session_id.
    """
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    # Get the session data from Redis
    session_data = redis_client.get_session_data(session_id)
    if not session_data:
        return Response(status_code=400, content="Session Expired or not found")

    print("Session data:", session_data)

    return {"file_name": session_data}


@router.get("/current-session")
async def get_session(session_id: str = Cookie(None)):
    if not session_id:
        return {"message": "No session found"}
    data = redis_client.get_session_data(session_id)
    if not data:
        return {"message": "Session expired or not found"}
    return {"session_data": data}


@router.post("/end-session")
async def end_session(session_id: str = Cookie(None), response: Response = None):
    if session_id:
        redis_client.delete_session_data(session_id)
        # Optionally clear the cookie
        response.delete_cookie(SESSION_COOKIE_NAME)
    return {"message": "Session ended"}
