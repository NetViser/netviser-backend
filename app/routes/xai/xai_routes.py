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

router = APIRouter(prefix="/api/attack-detection/xai", tags=["xai"])
settings = get_settings()
redis_client = RedisClient()


@router.get("/individual")
async def get_attack_detection_xai(
    attack_type: str,
    data_point_id: int,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Retrieve the data for XAI stored in S3 based on the session_id.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing")
    print("session_id", session_id)

    base_key = f"xai/{attack_type}/{data_point_id}"
    force_plot_image_key = f"{base_key}/images/force_plot.png"
    force_plot_text_key = f"{base_key}/texts/force_plot.text"

    # Check if both the image and SHAP text file already exist in S3.
    if await s3_service.file_exists(
        filename=force_plot_image_key, session_id=session_id
    ) and await s3_service.file_exists(
        filename=force_plot_text_key, session_id=session_id
    ):
        print("Both files exist in S3.", force_plot_image_key, force_plot_text_key)
        image_url = s3_service.get_url(
            s3_key=force_plot_image_key, session_id=session_id, expiration=21600
        )

        shap_txt_data = await s3_service.read(
            f"uploads/{session_id}/{force_plot_text_key}"
        )
        shap_text = shap_txt_data.decode("utf-8")
        return {"force_plot_url": image_url}

    print("pas the check")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=400, detail="Session Expired or not found")

        print("session_data", session_data)

        file_data = await s3_service.read(session_data)
        file_like_object = io.BytesIO(file_data)
        df = pd.read_csv(file_like_object, engine="pyarrow", dtype_backend="pyarrow")
        df.reset_index(inplace=True)

        # Load the pre-trained model and scaler
        model = xgb.Booster()
        model.load_model(os.path.join("model", "xgb_booster.model"))
        scaler = joblib.load(os.path.join("model", "scaler.pkl"))
        label_encoder = joblib.load(os.path.join("model", "label_encoder.pkl"))

        explainer = shap.TreeExplainer(model)
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

        # Scale the input features
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        # Prepare un-scaled data for SHAP values
        X_unscaled = df[feature_cols]

        # Single network flow data point
        single_instance_scaled = X_scaled_df.iloc[[data_point_id]]
        dtest_single = xgb.DMatrix(single_instance_scaled)

        # Get predicted class
        print("A1", df["Label"].iloc[data_point_id])
        predicted_class_index = label_encoder.transform(
            [df["Label"].iloc[data_point_id]]
        )[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

        print(f"Predicted class: {predicted_class}")

        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer.shap_values(single_instance_scaled)

        def format_val(x):
            return ("%.2f" % x).rstrip("0").rstrip(".")

        original_values_row = X_unscaled.iloc[data_point_id].copy()
        formatted_features = original_values_row.apply(format_val)

        attack_shap_val = shap_values_single[0, :, predicted_class_index]

        expected_value_for_class = explainer.expected_value[predicted_class_index]

        force_plot = shap.plots.force(
            base_value=expected_value_for_class,
            shap_values=attack_shap_val,
            features=formatted_features,
            feature_names=formatted_features.index.tolist(),
            matplotlib=True,  # Use the matplotlib-based plot
            figsize=(30, 6),
            show=False,
        )

        force_plot.figure.suptitle(
            f"Force Plot for Instance {data_point_id} (Pred: {predicted_class})",
            fontsize=16,
        )
        force_plot.figure.tight_layout()

        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        force_plot.figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_data = buf.getvalue()

        # Generate a text representation of the SHAP values (for example, a simple table)

        base_value = explainer.expected_value[predicted_class_index]
        class_name = label_encoder.inverse_transform([predicted_class_index])[0]

        # Create a DataFrame (force_data) with Feature, Feature Value, and SHAP Value
        shap_val = attack_shap_val
        force_data = pd.DataFrame(
            {
                "Feature": X_unscaled.columns,
                "Feature Value": formatted_features,
                "SHAP Value": shap_val,
            }
        )

        # Sort features by absolute SHAP value descending
        force_data = force_data.reindex(
            force_data["SHAP Value"].abs().sort_values(ascending=False).index
        )

        # Start building a text output
        shap_text = ""
        shap_text += f"--- Data Behind Force Plot for Class {predicted_class_index}: {class_name} ---\n"
        shap_text += f"Base (Expected) Value: {base_value}\n\n"

        # Add a simple table header
        shap_text += f"{'Feature':30s} | {'Feature Value':>12s} | {'SHAP Value':>10s}\n"
        shap_text += f"{'-'*30} | {'-'*12} | {'-'*10}\n"

        # Iterate over each feature row and format the columns
        for i, row in force_data.iterrows():
            feature_name = str(row["Feature"])
            feature_value = str(row["Feature Value"])
            shap_value = row["SHAP Value"]
            shap_text += (
                f"{feature_name:30s} | {feature_value:>12s} | {shap_value:>10.4f}\n"
            )

        # Example final print (for debugging). Then upload shap_text to S3 as before.
        print("\nGenerated SHAP Text:\n", shap_text)

        upload_force_plot_image_file = UploadFile(
            filename="force_plot.png", file=io.BytesIO(image_data)
        )
        upload_force_plot_text_file = UploadFile(
            filename="force_plot.text", file=io.BytesIO(shap_text.encode("utf-8"))
        )

        await s3_service.upload(
            file=upload_force_plot_image_file,
            file_path=force_plot_image_key,
            session_id=session_id,
        )
        await s3_service.upload(
            file=upload_force_plot_text_file,
            file_path=force_plot_text_key,
            session_id=session_id,
        )

        return {
            attack_type: attack_type,
            "data_point_id": data_point_id,
            "force_plot_url": s3_service.get_url(
                s3_key=force_plot_image_key, session_id=session_id, expiration=21600
            ),
        }
    except Exception as e:
        print("ERROR", e)
        return Response(status_code=400, content="Failed to retrieve data.")


@router.get("/individual/explanation")
async def get_attack_detection_xai_explanation(
    attack_type: str,
    data_point_id: int,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
    gemini_service: GeminiService = Depends(GeminiService),
):
    # Validate the session cookie
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing")

    # Retrieve session data from Redis
    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=400, detail="Session expired or not found")
    except Exception as exc:
        # Optionally log the exception: logger.error(f"Session retrieval error: {exc}")
        raise HTTPException(status_code=400, detail="Error retrieving session data")

    # Build the key for the SHAP force plot text file
    base_key = f"xai/{attack_type}/{data_point_id}"
    force_plot_text_key = f"{base_key}/texts/force_plot.text"

    # Check if the file exists in S3
    try:
        file_exists = await s3_service.file_exists(filename=force_plot_text_key, session_id=session_id)
    except Exception as exc:
        # Optionally log the exception: logger.error(f"S3 file existence check error: {exc}")
        raise HTTPException(status_code=500, detail="Error checking S3 for force plot text file")

    if not file_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Explanation not found for {attack_type} {data_point_id}.",
        )

    # Read the SHAP text file from S3
    try:
        shap_txt_data = await s3_service.read(f"uploads/{session_id}/{force_plot_text_key}")
        shap_text = shap_txt_data.decode("utf-8")
    except Exception as exc:
        # Optionally log the exception: logger.error(f"Error reading SHAP text file: {exc}")
        raise HTTPException(status_code=500, detail="Error reading SHAP text file from S3")

    # Generate the explanation using Gemini
    try:
        explanation = gemini_service.generate_shap_force_plot_explanation(shap_text)
    except Exception as exc:
        print("ERROR", exc)
        # Optionally log the exception: logger.error(f"Gemini API error: {exc}")
        raise HTTPException(status_code=500, detail="Error generating explanation from Gemini API")

    return {"explanation": explanation}
