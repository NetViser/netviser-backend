from collections import Counter
import io
import uuid

from app.services.lambda_service import LambdaService
from app.services.redis_service import RedisClient
from fastapi import (
    APIRouter,
    File,
    Form,
    Response,
    UploadFile,
    Cookie,
    Depends,
    HTTPException,
    Query,
)
from typing import Optional
from app.configs.config import get_settings
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
    filename: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None),  # Keep support for sample files
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Generate a presigned URL for the client to upload a file directly to S3.
    Supports sample files as an alternative.
    """
    # Create a new session
    session_id = str(uuid.uuid4())
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=43200,  # 12 hours
        httponly=True,
        secure=settings.SECURE_COOKIE,
        samesite=settings.SAMESITE,
    )

    try:
        if sample_filename:
            # Handle sample file case (no presigned URL needed)
            sample_mapping = {
                "ssh-ftp.csv": "sample/model-applied/ssh-ftp.csv",
                "ddos-ftp.csv": "sample/model-applied/ddos-ftp.csv",
                "ftp_patator_occurence.csv": "sample/model-applied/ftp_patator_occurence.csv",
                "portscan_dos_hulk_slowloris.csv": "sample/model-applied/portscan_dos_hulk_slowloris.csv",
                "portscan_dos_hulk.csv": "sample/model-applied/portscan_dos_hulk.csv",
                "portscan.csv": "sample/model-applied/portscan.csv",
            }
            model_applied_s3_key = sample_mapping.get(sample_filename)
            if not model_applied_s3_key:
                raise HTTPException(status_code=400, detail="Invalid sample file name")

            redis_client.set_session_data(
                session_id, model_applied_s3_key, ttl_in_seconds=43200
            )
            return {
                "completed": True,
                "message": "Sample file selected and stored in session",
                "session_id": session_id,
                "s3_key": model_applied_s3_key,
            }

        # Handle client-side upload with presigned URL
        raw_file_path = f"network-file/raw/{filename}"
        raw_s3_key = f"{s3_service.path_prefix}/{session_id}/{raw_file_path}"

        # Generate presigned URL for raw file upload
        presigned_url = await s3_service.generate_presigned_url(
            file_path=raw_file_path,  # Pass the file_path directly
            expiration=180,  # 3 minutes
            session_id=session_id,
        )

        return {
            "completed": False,
            "message": "Upload the file to the provided presigned URL",
            "session_id": session_id,
            "presigned_url": presigned_url,
            "raw_file_path": raw_s3_key,  # Full S3 key for the raw file
        }

    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate file upload.")


@router.post("/upload-complete")
async def upload_complete(
    session_id: str = Cookie(...),
    raw_file_path: str = Form(...),  # Full S3 key from /upload
    s3_service: S3 = Depends(S3),  # Inject S3 service
    lambda_service: LambdaService = Depends(LambdaService),
):
    """
    Trigger Lambda processing after the client uploads the file to S3.
    Checks if raw file exists and generates model_applied_file_path internally.
    """
    try:
        # Step 1: Check if the raw file exists in S3
        file_exists = await s3_service.object_exists(s3_key=raw_file_path)
        if not file_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Raw file not found in S3 at path: {raw_file_path}",
            )

        # Step 2: Generate a unique model_applied_file_path
        raw_filename = raw_file_path.split("/")[-1]  # Extract filename from path
        model_applied_file_path = "network-file/model-applied/" + raw_filename
        model_applied_s3_key = (
            f"{s3_service.path_prefix}/{session_id}/{model_applied_file_path}"
        )

        # Step 3: Prepare Lambda payload
        lambda_service_payload = {
            "data": {
                "raw_file_path": raw_file_path,
                "model_applied_file_path": model_applied_s3_key,
            }
        }

        # Step 4: Invoke Lambda function for processing
        lambda_inference_output = await lambda_service.invoke_function(
            function_name="inference_func",
            function_params=lambda_service_payload,
        )

        # Step 5: Get the S3 key of the processed file from Lambda response
        model_applied_s3_key = lambda_inference_output.get("file_key")
        if not model_applied_s3_key:
            raise HTTPException(
                status_code=500, detail="Lambda did not return a valid file key"
            )

        # Step 6: Update session data in Redis with the processed file key
        redis_client.set_session_data(
            session_id, model_applied_s3_key, ttl_in_seconds=43200
        )

        return {
            "message": "File processed successfully",
            "session_id": session_id,
            "s3_key": model_applied_s3_key,
        }

    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")


@router.post("/upload-legacy")
async def upload_file_legacy(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    file: Optional[UploadFile] = File(None),
    samplefile: Optional[str] = None,
    s3_service: S3 = Depends(S3),
    lambda_service: LambdaService = Depends(LambdaService),
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
    # Determine the maximum upload size (default to 10MB if not defined)
    MAX_UPLOAD_SIZE = getattr(
        settings, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024
    )  # 10MB in bytes
    print(f"MAX_UPLOAD_SIZE: {MAX_UPLOAD_SIZE}")

    # 1. Validate input file
    print(f"Sample file: {samplefile}")
    if not samplefile:
        if not file:
            raise ValueError("File missing")

        # Check file size before processing
        file.file.seek(0, 2)  # Move to the end of the file to get size
        file_size = file.file.tell()
        file.file.seek(0)  # Reset file pointer to the beginning
        print(f"File size: {file_size}")

        if file_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the maximum limit of {MAX_UPLOAD_SIZE / (1024 * 1024)} MB.",
            )

    # 2. Create a new session
    session_id = str(uuid.uuid4())
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=43200,  # 12 hours
        httponly=True,
        secure=settings.SECURE_COOKIE,
        samesite=settings.SAMESITE,
    )

    try:
        if not samplefile:
            raw_file = UploadFile(filename=f"{file.filename}", file=file.file)

            raw_file_info = await s3_service.upload(
                file=raw_file,
                file_path=f"network-file/raw/{raw_file.filename}",
                session_id=session_id,
            )

            lambda_service_payload = {
                "data": {
                    "raw_file_path": raw_file_info.get("s3_key"),
                    "model_applied_file_path": f"uploads/{session_id}/network-file/model-applied/{raw_file.filename}",
                }
            }

            lambda_inference_output = await lambda_service.invoke_function(
                function_name="inference_func",
                function_params=lambda_service_payload,
            )

            model_applied_s3_key = lambda_inference_output.get("file_key")

            return_msg = "DataFrame successfully processed and stored in session."

        else:
            sample_mapping = {
                "ssh-ftp.csv": "sample/model-applied/ssh-ftp.csv",
                "ddos-ftp.csv": "sample/model-applied/ddos-ftp.csv",
                "ftp_patator_occurence.csv": "sample/model-applied/ftp_patator_occurence.csv",
                "portscan_dos_hulk_slowloris.csv": "sample/model-applied/portscan_dos_hulk_slowloris.csv",
                "portscan_dos_hulk.csv": "sample/model-applied/portscan_dos_hulk.csv",
                "portscan.csv": "sample/model-applied/portscan.csv",
            }

            model_applied_s3_key = sample_mapping.get(samplefile)

            return_msg = "Sample file successfully processed and stored in session."

        # Store or update the session data in Redis with the processed file's S3 key
        redis_client.set_session_data(
            session_id, model_applied_s3_key, ttl_in_seconds=43200
        )

        return {
            "content": {
                "message": return_msg,
                "session_id": session_id,
                "s3_key": model_applied_s3_key,
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
