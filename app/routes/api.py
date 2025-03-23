from collections import Counter
import io
import uuid
import pandas as pd
import logging
from app.schemas.upload import UploadCompleteResponse, UploadPresignedResponse, UploadSampleResponse
from app.services.model_service import get_model_artifacts, predict_df
from app.services.redis_service import RedisClient
from fastapi import (
    APIRouter,
    Form,
    Response,
    UploadFile,
    Cookie,
    Depends,
    HTTPException,
    Query,
)

from typing import Optional, Union
from app.configs.config import get_settings
from app.services.bucket_service import GCS
from app.services.input_handle_service import preprocess
from app.services.model_service import (
    feature_columns,
    preprocess_df,
    predict_df,
    get_model_artifacts,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api", tags=["api"])
settings = get_settings()

# Since we don't have real login, we'll store a custom "session_id" in a cookie
SESSION_COOKIE_NAME = "session_id"

# Instantiate our singleton Redis client once
redis_client = RedisClient()


@router.get("/dashboard")
async def get_dashboard(
    session_id: Optional[str] = Cookie(None),
    bucket_service: GCS = Depends(GCS),
):
    """
    Retrieve the data for dashboard stored in GCS and preprocess it.
    """
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            return Response(status_code=400, content="Session Expired or not found")

        file_data = await bucket_service.read(session_data)
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
            data_frame["flow_bytes/s"].resample("1s").mean().reset_index()
        )
        flow_bytes_resampled.columns = ["timestamp", "value"]

        flow_bytes_resampled["value"] = flow_bytes_resampled["value"].round(2)
        flow_bytes_list = flow_bytes_resampled.dropna().to_dict(orient="records")

        # Resample Forward Packets/s
        fwd_packets_resampled = (
            data_frame["fwd_packets/s"].resample("1s").mean().reset_index()
        )
        fwd_packets_resampled.columns = ["timestamp", "value"]

        fwd_packets_resampled["value"] = fwd_packets_resampled["value"].round(2)
        fwd_packets_list = fwd_packets_resampled.dropna().to_dict(orient="records")

        # Resample Backward Packets/s
        bwd_packets_resampled = (
            data_frame["bwd_packets/s"].resample("1s").mean().reset_index()
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


@router.get("/attack-detection/records")
async def fetch_attack_records(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    page: int = Query(1, ge=1),  # Default to page 1, must be >= 1
    page_size: int = Query(10, ge=1, le=100),  # Default to 10, max 100 per page
    bucket_service: GCS = Depends(GCS),
):
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            return Response(status_code=400, content="Session Expired or not found")

        file_data = await bucket_service.read(session_data)
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


@router.post("/upload", response_model=Union[UploadSampleResponse, UploadPresignedResponse])
async def upload_file(
    response: Response,
    filename: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None),
    bucket_service: GCS = Depends(GCS),
):
    """
    Generate a presigned URL for the client to upload a file directly to GCS.
    Supports sample files as an alternative.
    """
    logger.debug("Entering /upload endpoint")
    logger.info(
        f"Received request with filename: {filename}, sample_filename: {sample_filename}, session_id: {session_id}"
    )

    # Create a new session
    session_id = str(uuid.uuid4())
    logger.debug(f"Generated new session_id: {session_id}")

    try:
        logger.info("Setting session cookie")
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=43200,
            httponly=True,
            secure=settings.SECURE_COOKIE,
            samesite=settings.SAMESITE,
        )
        logger.debug("Session cookie set successfully")

        if sample_filename:
            logger.info(f"Processing sample file: {sample_filename}")
            sample_mapping = {
                "ssh-ftp.csv": "sample/model-applied/ssh-ftp.csv",
                "ddos-ftp.csv": "sample/model-applied/ddos-ftp.csv",
                "ftp_patator_occurence.csv": "sample/model-applied/ftp_patator_occurence.csv",
                "portscan_dos_hulk_slowloris.csv": "sample/model-applied/portscan_dos_hulk_slowloris.csv",
                "portscan_dos_hulk.csv": "sample/model-applied/portscan_dos_hulk.csv",
                "portscan.csv": "sample/model-applied/portscan.csv",
            }

            model_applied_gcs_key = sample_mapping.get(sample_filename)
            if not model_applied_gcs_key:
                logger.warning(f"Invalid sample file name: {sample_filename}")
                raise HTTPException(status_code=400, detail="Invalid sample file name")

            logger.debug(f"Mapping found: {model_applied_gcs_key}")
            logger.info(f"Setting Redis session data for sample file")
            redis_client.set_session_data(
                session_id, model_applied_gcs_key, ttl_in_seconds=43200
            )
            logger.debug("Redis session data set successfully")

            response_data = {
                "completed": True,
                "message": "Sample file selected and stored in session",
                "session_id": session_id,
                "bucket_key": model_applied_gcs_key,
            }
            logger.info(f"Returning successful sample file response: {response_data}")
            return response_data

        # Handle client-side upload with presigned URL
        logger.info("Processing regular file upload")
        if not filename:
            logger.warning("No filename provided for regular upload")
            raise HTTPException(status_code=400, detail="Filename required for upload")

        raw_file_path = f"network-file/raw/{filename}"
        raw_gcs_key = f"{bucket_service.path_prefix}/{session_id}/{raw_file_path}"
        logger.debug(f"Generated raw file path: {raw_file_path}")
        logger.debug(f"Generated raw GCS key: {raw_gcs_key}")

        logger.info("Generating presigned URL")
        presigned_url = await bucket_service.generate_presigned_url(
            file_path=raw_file_path,
            expiration=180,
            session_id=session_id,
        )
        logger.debug(f"Presigned URL generated: {presigned_url}")

        response_data = {
            "completed": False,
            "message": "Upload the file to the provided presigned URL",
            "session_id": session_id,
            "presigned_url": presigned_url,
            "bucket_key": raw_gcs_key,
        }
        logger.info(f"Returning presigned URL response: {response_data}")
        return response_data

    except HTTPException as he:
        logger.error(f"HTTPException occurred: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initiate file upload.")


@router.post("/upload-complete", response_model=UploadCompleteResponse)
async def upload_complete(
    session_id: str = Cookie(...),
    raw_file_path: str = Form(...),
    bucket_service: GCS = Depends(GCS),
):
    """
    Trigger Lambda processing after the client uploads the file to GCS.
    Checks if raw file exists and generates model_applied_file_path internally.
    """
    try:
        file_exists = await bucket_service.object_exists(gcs_key=raw_file_path)
        if not file_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Raw file not found in GCS at path: {raw_file_path}",
            )

        raw_filename = raw_file_path.split("/")[-1]  # Extract filename from path
        model_applied_file_path = "network-file/model-applied/" + raw_filename

        model_applied_gcs_key = (
            f"{bucket_service.path_prefix}/{session_id}/{model_applied_file_path}"
        )

        file_bytes = await bucket_service.read(raw_file_path)
        file_like_object = io.BytesIO(file_bytes)

        if file_like_object is None:
            raise HTTPException(
                status_code=500, detail="Failed to read the uploaded file."
            )

        df = pd.read_csv(file_like_object, engine="pyarrow", dtype_backend="pyarrow")
        df_processed: pd.DataFrame = preprocess_df(df)

        if (df_processed.empty) or (df_processed is None):
            raise HTTPException(
                status_code=500, detail="Failed to preprocess the uploaded file."
            )

        model, scaler, label_encoder = get_model_artifacts()
        predicted_labels = predict_df(
            df_processed[feature_columns], model, scaler, label_encoder
        )
        df_processed["Label"] = predicted_labels

        labeled_buffer = io.BytesIO()
        df_processed.to_csv(labeled_buffer, index=False)
        labeled_buffer.seek(0)
        labeled_file = UploadFile(filename=model_applied_file_path, file=labeled_buffer)

        await bucket_service.upload(
            file=labeled_file,
            file_path=model_applied_file_path,
            session_id=session_id,
        )

        redis_client.set_session_data(
            session_id, model_applied_gcs_key, ttl_in_seconds=43200
        )

        return {
            "message": "File processed successfully",
            "session_id": session_id,
            "bucket_key": model_applied_gcs_key,
        }

    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")


@router.get("/attack-detection/brief/scatter")
async def get_attack_detection_brief_scatter(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    bucket_service: GCS = Depends(GCS),
):
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            return Response(status_code=400, content="Session Expired or not found")

        file_data = await bucket_service.read(session_data)
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

    except Exception:
        return Response(status_code=400, content="Failed to retrieve data.")


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
