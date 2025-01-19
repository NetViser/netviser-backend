from collections import Counter
import io
import uuid

import numpy as np
from app.services.redis_service import RedisClient
from fastapi import (
    APIRouter,
    File,
    Response,
    UploadFile,
    Cookie,
    Depends,
    HTTPException,
)
from typing import Optional
from app.configs.config import get_settings
import xgboost as xgb
import pandas as pd
import os
import joblib
from app.services.s3_service import S3
from app.services.input_handle_service import preprocess


router = APIRouter(prefix="/api", tags=["api"])
settings = get_settings()

# Since we don't have real login, we'll store a custom "session_id" in a cookie
SESSION_COOKIE_NAME = "session_id"

# Instantiate our singleton Redis client once
redis_client = RedisClient()

@router.get("/dashboard")
async def get_dashboard(
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Retrieve the data for dashboard stored in S3 based on the session_id.
    """
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            return Response(status_code=400, content="Session Expired or not found")

        file_data = await s3_service.read(session_data)
        file_like_object = io.BytesIO(file_data)

        data_frame = await preprocess(file_like_object)

        # Compute counts and distributions
        benign_count = data_frame[data_frame["Label"] == "BENIGN"].shape[0]
        total_rows = len(data_frame)
        non_benign_count = total_rows - benign_count


        # Rename columns for resampling
        data_frame.rename(columns={
            "Flow Bytes/s": "flow_bytes/s",
            "Fwd Packets/s": "fwd_packets/s",
            "Bwd Packets/s": "bwd_packets/s"
        }, inplace=True)

        # Resample Flow Bytes/s
        flow_bytes_resampled = (
            data_frame["flow_bytes/s"]
            .resample("1S")
            .mean()
            .reset_index()
        )
        flow_bytes_resampled.columns = ["timestamp", "value"]

        flow_bytes_resampled["value"] = flow_bytes_resampled["value"].round(2)
        flow_bytes_list = flow_bytes_resampled.dropna().to_dict(orient="records")

        # Resample Forward Packets/s
        fwd_packets_resampled = (
            data_frame["fwd_packets/s"]
            .resample("1S")
            .mean()
            .reset_index()
        )
        fwd_packets_resampled.columns = ["timestamp", "value"]

        fwd_packets_resampled["value"] = fwd_packets_resampled["value"].round(2)
        fwd_packets_list = fwd_packets_resampled.dropna().to_dict(orient="records")

        # Resample Backward Packets/s
        bwd_packets_resampled = (
            data_frame["bwd_packets/s"]
            .resample("1S")
            .mean()
            .reset_index()
        )
        bwd_packets_resampled.columns = ["timestamp", "value"]

        bwd_packets_resampled["value"] = bwd_packets_resampled["value"].round(2)
        bwd_packets_list = bwd_packets_resampled.dropna().to_dict(orient="records")

        # Calculate distributions
        src_port_distribution = Counter(data_frame["Src Port"])
        dst_port_distribution = Counter(data_frame["Dst Port"])
        protocol_distribution = Counter(data_frame["Protocol"])
        src_ip_address_distribution = Counter(data_frame["Src IP"])
        label_distribution = Counter(data_frame["Label"])

        if "BENIGN" in label_distribution:
            del label_distribution["BENIGN"]

        # Reset index so further operations can use columns normally
        data_frame.reset_index(inplace=True)

        return {
            "file_name": session_data,
            "total_rows": total_rows,
            "total_detected_attacks": non_benign_count,
            "detected_attacks_distribution": dict(label_distribution),
            "src_ip_address_distribution": dict(src_ip_address_distribution),
            "src_port_distribution": dict(src_port_distribution),
            "dst_port_distribution": dict(dst_port_distribution),
            "protocol_distribution": dict(protocol_distribution),
            "flow_bytes_per_second": flow_bytes_list,
            "fwd_packets_per_second": fwd_packets_list,
            "bwd_packets_per_second": bwd_packets_list
        }

    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve dashboard.")


@router.post("/upload")
async def upload_file(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    file: UploadFile = File(...),
    s3_service: S3 = Depends(S3),
):
    """
    Store file in S3 as a pickle.
    """

    # 1. Validate input file
    if not file:
        raise ValueError("file missing")

    if not session_id:
        # If the user doesn't have a session_id, create a new one
        session_id = str(uuid.uuid4())
        # Set the cookie with a 5-minute expiration
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=43200,
            httponly=True,  # Prevents JavaScript access
            secure=(
                False if settings.STAGE == "local" else True
            ),  # Set to True in production
            samesite="lax",  # Adjust as needed
        )

    try:
        # 3. Read CSV into a pandas DataFrame
        df = pd.read_csv(file.file, engine="pyarrow", dtype_backend="pyarrow")

        meta_columns = ["Flow ID", "Timestamp", "Src IP", "Dst IP"]
        # Make a copy to avoid altering these columns inadvertently
        meta_df = df[meta_columns].copy()

        # 4. Drop irrelevant columns
        irrelevant_columns = [
            "id",
            "Flow ID",
            "Attempted Category",
            "Timestamp",
            "Src IP",
            "Dst IP",
            "Hour",
            "Day",
        ]
        df.drop(columns=irrelevant_columns, axis=1, inplace=True, errors="ignore")

        # 5. Clean data (handle inf/nan)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)

        # 6. Load pre-trained XGBoost model and scaler
        model = xgb.Booster()
        load_model = os.path.join("model", "xgb_booster.model")
        model.load_model(load_model)

        scaler_path = os.path.join("model", "scaler.pkl")
        scaler = joblib.load(scaler_path)

        # 7. Define features and transform
        feature_cols = [
            "Src Port",
            "Dst Port",
            "Total TCP Flow Time",
            "Bwd Init Win Bytes",
            "Bwd Packet Length Std",
            "Total Length of Fwd Packet",
            "Fwd Packet Length Max",
            "Bwd IAT Mean",
            "Flow IAT Min",
            "Fwd PSH Flags",
        ]
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        dmatrix = xgb.DMatrix(X_scaled)

        # 8. Predict classes
        #    If your model outputs class probabilities (shape [n_samples, n_classes]),
        #    then np.argmax along axis=1 is the typical way to get the class index.
        predictions = model.predict(dmatrix)
        predicted_indices = np.argmax(predictions, axis=1)

        # 9. Decode labels back to their string form
        label_encoder_path = os.path.join("model", "label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        decoded_labels = label_encoder.inverse_transform(predicted_indices)

        df["Label"] = decoded_labels
        df = pd.concat([meta_df, df], axis=1)

        # 10. Convert the DataFrame to a pickle in memory
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        # 11. Create an UploadFile-like object for S3 upload
        #     We'll call this `processed_file` instead of `csv_file`
        processed_file = UploadFile(
            filename=f"{file.filename}_processed.csv", file=buffer
        )

        # 12. Upload the pickle file to S3
        upload_output = await s3_service.upload(
            processed_file, processed_file.filename, session_id
        )
        s3_key = upload_output.get("s3_key")

        # 13. Store or update the session data in Redis with a 10-minute TTL
        redis_client.set_session_data(session_id, s3_key, ttl_in_seconds=43200)

        return {
            "content": {
                "message": "DataFrame successfully processed and stored in session.",
                "session_id": session_id,
                "s3_key": s3_key,
            },
            "status_code": 200,
        }

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to process the uploaded file."
        )


@router.get("/attack-detection/brief/scatter")
async def get_attack_detection_brief_scatter(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            return Response(status_code=400, content="Session Expired or not found")

        file_data = await s3_service.read(session_data)
        file_like_object = io.BytesIO(file_data)

        data_frame = await preprocess(file_like_object)

        data_frame["is_attack"] = data_frame["Label"].apply(lambda x: x != "BENIGN")

        data_frame.reset_index(inplace=True)

        data_frame["Flow Bytes/s"] = data_frame["Flow Bytes/s"].astype(float)

        # Round to two decimal places
        data_frame["Flow Bytes/s"] = data_frame["Flow Bytes/s"].round(2)

        benign_data = [
            {
                "timestamp": row["Timestamp"].isoformat(),
                "value": row["Flow Bytes/s"]
            }
            for row in data_frame[data_frame["is_attack"] == False].dropna().to_dict(orient="records")
        ]

        attack_data = [
            {
                "timestamp": row["Timestamp"].isoformat(),
                "value": row["Flow Bytes/s"]
            }
            for row in data_frame[data_frame["is_attack"] == True].dropna().to_dict(orient="records")
        ]

        return {
            "attack_type": attack_type,
            "benign_data": benign_data,
            "attack_data": attack_data,
            "feature_name": "Flow Bytes/s",
        }

    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve data.")


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
            httponly=True,  # Prevents JavaScript access
            secure=(False if settings.STAGE == "local" else True),
            samesite="lax",  # Adjust as needed
        )

    try:
        # Upload the raw file to S3
        upload_output = await s3_service.upload(file, file.filename, session_id)
        print("upload output:", upload_output)
        s3_key = upload_output.get("s3_key")
        print(f"File uploaded to S3 with key: {s3_key}")

        # Store or update the session data in Redis with a 5-minute TTL
        redis_client.set_session_data(session_id, s3_key, ttl_in_seconds=19960)
        print("Session data stored in Redis.")
        return {
            "message": "File successfully stored in session.",
            "session_id": session_id,
        }

    except Exception as e:
        print(e)
        return {
            "message": "Failed to process the uploaded file.",
        }


@router.get("/get-file-name")
async def get_file_name(session_id: str = Cookie(None)):
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
