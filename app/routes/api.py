import asyncio
from collections import Counter
import io
import uuid

import numpy as np
from app.services.gemini_service import GeminiService
from app.services.redis_service import RedisClient
from fastapi import (
    APIRouter,
    File,
    Response,
    UploadFile,
    Cookie,
    Depends,
    HTTPException,
    Query,
)
from typing import Optional
from app.configs.config import get_settings
import xgboost as xgb
import pandas as pd
import os
import joblib
import shap
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
        data_frame.rename(
            columns={
                "Flow Bytes/s": "flow_bytes/s",
                "Fwd Packets/s": "fwd_packets/s",
                "Bwd Packets/s": "bwd_packets/s",
            },
            inplace=True,
        )

        # Resample Flow Bytes/s
        flow_bytes_resampled = (
            data_frame["flow_bytes/s"].resample("1S").mean().reset_index()
        )
        flow_bytes_resampled.columns = ["timestamp", "value"]

        flow_bytes_resampled["value"] = flow_bytes_resampled["value"].round(2)
        flow_bytes_list = flow_bytes_resampled.dropna().to_dict(orient="records")

        # Resample Forward Packets/s
        fwd_packets_resampled = (
            data_frame["fwd_packets/s"].resample("1S").mean().reset_index()
        )
        fwd_packets_resampled.columns = ["timestamp", "value"]

        fwd_packets_resampled["value"] = fwd_packets_resampled["value"].round(2)
        fwd_packets_list = fwd_packets_resampled.dropna().to_dict(orient="records")

        # Resample Backward Packets/s
        bwd_packets_resampled = (
            data_frame["bwd_packets/s"].resample("1S").mean().reset_index()
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
            "bwd_packets_per_second": bwd_packets_list,
        }

    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve dashboard.")


@router.get("/attack-detection/specific")
async def get_specific_attack_detection(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Retrieve the data for a specific attack type stored in S3 based on the session_id.
    Paginated results are returned for both normal and attack data.
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

        data_frame.reset_index(inplace=True)

        # Round specified fields to 2 decimal places and apply log scale
        fields_to_round = [
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow Duration",
            "Average Packet Size",
        ]
        for field in fields_to_round:
            data_frame[field] = data_frame[field].astype(float).round(2)
            data_frame[field] = np.log10(data_frame[field] + 1)

        # Count the number of occurrences for each Src Port and Dst Port pair
        port_pair_counts = (
            data_frame.groupby(["Src Port", "Dst Port"])
            .size()
            .reset_index(name="Port Pair Count")
        )

        # Count the number of occurrences for each Src IP and Src Port pair
        src_ip_port_pair_counts = (
            data_frame.groupby(["Src IP", "Src Port"])
            .size()
            .reset_index(name="Src IP Port Pair Count")
        )

        # Merge the counts back into the original DataFrame
        data_frame = data_frame.merge(
            port_pair_counts, on=["Src Port", "Dst Port"], how="left"
        )

        # Merge the Src IP Port Pair Count back into the original DataFrame
        data_frame = data_frame.merge(
            src_ip_port_pair_counts, on=["Src IP", "Src Port"], how="left"
        )

        # Separate data into normal and attack DataFrames
        normal_df = data_frame[data_frame["Label"] == "BENIGN"]
        normal_df = normal_df.sort_values(by="Timestamp", ascending=False)
        attack_df = data_frame[data_frame["Label"] == attack_type]
        attack_df = attack_df.sort_values(by="Timestamp", ascending=False)

        # Prepare normal data in camelCase
        normal_data = [
            {
                "timestamp": row["Timestamp"].isoformat(),
                "flowBytesPerSecond": row["Flow Bytes/s"],
                "flowDuration": row["Flow Duration"],
                "flowPacketsPerSecond": row["Flow Packets/s"],
                "averagePacketSize": row["Average Packet Size"],
                "totalFwdPacket": row["Total Fwd Packet"],
                "totalLengthOfFwdPacket": row["Total Length of Fwd Packet"],
                "protocol": row["Protocol"],
                "srcIp": row["Src IP"],
                "dstIp": row["Dst IP"],
                "srcPort": row["Src Port"],
                "dstPort": row["Dst Port"],
                "portPairCount": row["Port Pair Count"],
                "srcIpPortPairCount": row["Src IP Port Pair Count"],
            }
            for row in normal_df.dropna().to_dict(orient="records")
        ]

        # Prepare attack data in camelCase
        attack_data = [
            {
                "timestamp": row["Timestamp"].isoformat(),
                "flowBytesPerSecond": row["Flow Bytes/s"],
                "flowDuration": row["Flow Duration"],
                "flowPacketsPerSecond": row["Flow Packets/s"],
                "averagePacketSize": row["Average Packet Size"],
                "totalFwdPacket": row["Total Fwd Packet"],
                "totalLengthOfFwdPacket": row["Total Length of Fwd Packet"],
                "protocol": row["Protocol"],
                "srcIp": row["Src IP"],
                "dstIp": row["Dst IP"],
                "srcPort": row["Src Port"],
                "dstPort": row["Dst Port"],
                "portPairCount": row["Port Pair Count"],
                "srcIpPortPairCount": row["Src IP Port Pair Count"],
            }
            for row in attack_df.dropna().to_dict(orient="records")
        ]

        return {
            "normalData": normal_data,
            "attackData": attack_data,
        }

    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve data.")


@router.get("/attack-detection/records")
async def fetch_attack_records(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    page: int = Query(1, ge=1),  # Default to page 1, must be >= 1
    page_size: int = Query(10, ge=1, le=100),  # Default to 10, max 100 per page
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
        attack_df = data_frame[data_frame["Label"] == attack_type]
        attack_df.reset_index(inplace=True)

        attack_df = attack_df.sort_values(by="Timestamp", ascending=False)

        total_records = len(attack_df)
        total_pages = (total_records + page_size - 1) // page_size  # Ceiling division

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_attack_df = attack_df.iloc[start_idx:end_idx]

        attack_data = [
            {
                "id": row["id"],
                "timestamp": row["Timestamp"].isoformat(),
                "flowBytesPerSecond": row["Flow Bytes/s"],
                "flowDuration": row["Flow Duration"],
                "flowPacketsPerSecond": row["Flow Packets/s"],
                "avgPacketSize": row["Average Packet Size"],
                "totalFwdPacket": row["Total Fwd Packet"],
                "totalLengthFwdPacket": row["Total Length of Fwd Packet"],
                "protocol": row["Protocol"],
                "srcIP": row["Src IP"],
                "dstIP": row["Dst IP"],
                "srcPort": row["Src Port"],
                "dstPort": row["Dst Port"],
            }
            for row in paginated_attack_df.dropna().to_dict(orient="records")
        ]

        return {
            "attack_type": attack_type,
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
            "has_next_page": page < total_pages,
            "has_previous_page": page > 1,
            "next_page": page + 1 if page < total_pages else None,
            "previous_page": page - 1 if page > 1 else None,
            "attack_data": attack_data,
        }
    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve data.")


@router.post("/upload")
async def upload_file(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    file: UploadFile = File(...),
    s3_service: S3 = Depends(S3),
):
    """
    Process the uploaded CSV file by:
      1. Reading CSV data into a DataFrame.
      2. Adding a unique 'id' column.
      3. Cleaning data and dropping unnecessary columns.
      4. Predicting labels using a pre-trained XGBoost model.
      5. Uploading both the raw file and processed CSV to S3 in separate folders.
      6. Storing the S3 key of the processed file in the Redis session.
    """
    # 1. Validate input file
    if not file:
        raise ValueError("File missing")

    # 2. Create a new session if necessary
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=43200,  # 12 hours
            httponly=True,
            secure=(False if settings.STAGE == "local" else True),
            samesite="lax",
        )

    try:
        # 3. Read CSV into a DataFrame using pyarrow for speed/efficiency
        df = pd.read_csv(file.file, engine="pyarrow", dtype_backend="pyarrow")

        # 4. Add a unique 'id' column (using the DataFrame's index)
        df.reset_index(drop=True, inplace=True)
        df["id"] = df.index

        # 5. Extract meta columns to preserve key identifiers
        meta_columns = ["id", "Flow ID", "Timestamp", "Src IP", "Dst IP"]
        meta_df = df[meta_columns].copy()

        # 6. Drop columns not needed for model prediction.
        #    (We want to keep 'id' so that we can reference rows later.)
        irrelevant_columns = [
            "Flow ID",
            "Attempted Category",
            "Timestamp",
            "Src IP",
            "Dst IP",
            "Hour",
            "Day",
        ]
        df.drop(columns=irrelevant_columns, axis=1, inplace=True, errors="ignore")

        # 7. Clean data: replace infinities and drop any rows with missing values.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)

        # 8. Load pre-trained model and scaler
        model = xgb.Booster()
        model.load_model(os.path.join("model", "xgb_booster.model"))
        scaler = joblib.load(os.path.join("model", "scaler.pkl"))

        # 9. Define features that the model expects
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

        # 10. Predict class probabilities and decode the labels
        predictions = model.predict(dmatrix)
        predicted_indices = np.argmax(predictions, axis=1)
        label_encoder_path = os.path.join("model", "label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        decoded_labels = label_encoder.inverse_transform(predicted_indices)

        # 11. Attach the predicted labels to the DataFrame
        df["Label"] = decoded_labels

        # 12. Re-combine meta data with the processed DataFrame
        processed_df = pd.concat([meta_df, df], axis=1)
        print(f"Processed DataFrame shape:\n{processed_df.head(10)}")

        # 13. Write the processed DataFrame to an in-memory CSV buffer
        processed_buffer = io.BytesIO()
        processed_df.to_csv(processed_buffer, index=False)
        processed_buffer.seek(0)

        # 14. Prepare UploadFile objects for raw and processed files.
        #    Note: After reading the CSV, the file pointer is at the end,
        #          so we must reset it to 0 to capture the raw file content.
        file.file.seek(0)
        raw_file = UploadFile(filename=f"{file.filename}", file=file.file)
        processed_file = UploadFile(filename=f"{file.filename}", file=processed_buffer)

        # Create two tasks for parallel uploads
        upload_tasks = [
            s3_service.upload(
                file=raw_file,
                file_path=f"network-file/raw/{raw_file.filename}",
                session_id=session_id,
            ),
            s3_service.upload(
                file=processed_file,
                file_path=f"network-file/model-applied/{processed_file.filename}",
                session_id=session_id,
            ),
        ]

        # Run both upload tasks concurrently
        _, processed_upload_output = await asyncio.gather(*upload_tasks)

        # Retrieve the S3 key from the processed file upload
        s3_key = processed_upload_output.get("s3_key")

        # 17. Store or update the session data in Redis with the processed file's S3 key
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
            {"timestamp": row["Timestamp"].isoformat(), "value": row["Flow Bytes/s"]}
            for row in data_frame[~data_frame["is_attack"]]
            .dropna()
            .to_dict(orient="records")
        ]

        attack_data = [
            {"timestamp": row["Timestamp"].isoformat(), "value": row["Flow Bytes/s"]}
            for row in data_frame[data_frame["is_attack"]]
            .dropna()
            .to_dict(orient="records")
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
