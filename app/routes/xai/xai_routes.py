import asyncio
from collections import Counter
import io
import uuid

import numpy as np
from app.services.gemini_service import GeminiService
from app.services.redis_service import RedisClient
from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    HTTPException,
    Response,
    UploadFile,
    status,
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
from app.services.xai_service import compute_beeswarm_jitter, normalize_feature

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
    force_plot_image_key = f"{base_key}/force_plot/image.png"
    force_plot_text_key = f"{base_key}/force_plot/value.text"

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

    # Build the keys for the force plot text file and the Gemini explanation file
    base_key = f"xai/{attack_type}/{data_point_id}"
    force_plot_text_key = f"{base_key}/force_plot/value.text"
    gemini_explanation_key = f"{base_key}/force_plot/explanation.text"

    # Check if the SHAP force plot text file exists in S3
    try:
        file_exists = await s3_service.file_exists(
            filename=force_plot_text_key, session_id=session_id
        )
    except Exception as exc:
        # Optionally log the exception: logger.error(f"S3 file existence check error: {exc}")
        raise HTTPException(
            status_code=500, detail="Error checking S3 for force plot text file"
        )

    if not file_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Force plot text not found for {attack_type} {data_point_id}.",
        )

    # Check if the Gemini explanation file already exists in S3; if so, return it directly.
    try:
        explanation_exists = await s3_service.file_exists(
            filename=gemini_explanation_key, session_id=session_id
        )
    except Exception as exc:
        # Optionally log the exception: logger.error(f"S3 explanation file check error: {exc}")
        raise HTTPException(
            status_code=500, detail="Error checking S3 for explanation file"
        )

    if explanation_exists:
        try:
            explanation_data = await s3_service.read(
                f"uploads/{session_id}/{gemini_explanation_key}"
            )
            explanation_text = explanation_data.decode("utf-8")
            return {"explanation": explanation_text}
        except Exception as exc:
            # Optionally log the exception: logger.error(f"Error reading explanation file: {exc}")
            raise HTTPException(
                status_code=500, detail="Error reading explanation file from S3"
            )

    # Read the SHAP force plot text file from S3
    try:
        shap_txt_data = await s3_service.read(
            f"uploads/{session_id}/{force_plot_text_key}"
        )
        shap_text = shap_txt_data.decode("utf-8")
    except Exception as exc:
        # Optionally log the exception: logger.error(f"Error reading SHAP text file: {exc}")
        raise HTTPException(
            status_code=500, detail="Error reading SHAP text file from S3"
        )

    # Generate the explanation using Gemini
    try:
        explanation = gemini_service.generate_shap_force_plot_explanation(
            shap_text, attack_type
        )
    except Exception as exc:
        # Optionally log the exception: logger.error(f"Gemini API error: {exc}")
        raise HTTPException(
            status_code=500, detail="Error generating explanation from Gemini API"
        )

    # Save the generated explanation to S3 for future reuse
    try:
        from fastapi import UploadFile  # Ensure UploadFile is imported
        import io

        upload_explanation_file = UploadFile(
            filename="explanation.text", file=io.BytesIO(explanation.encode("utf-8"))
        )
        await s3_service.upload(
            file=upload_explanation_file,
            file_path=gemini_explanation_key,
            session_id=session_id,
        )
    except Exception as exc:
        # Optionally log the exception: logger.error(f"Error saving explanation file: {exc}")
        # Do not block the response if saving fails; simply log the error.
        print("Warning: Failed to save explanation file to S3:", exc)

    return {"explanation": explanation}


@router.get("/summary")
async def get_attack_detection_xai_summary(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Retrieve or generate a SHAP bar-summary CSV for the specified attack_type
    and session_id from S3. If the CSV does not exist, compute and upload it.
    The heavy computation is offloaded to a worker thread to improve concurrency.
    """
    # Validate session_id
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Session ID missing"
        )

    # Retrieve session data from Redis
    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session expired or not found",
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error retrieving session data",
        )

    # Check if both CSVs exist in S3. If yes, return them directly.
    base_key = f"xai/{attack_type}/summary"
    bar_summary_key = f"{base_key}/bar_summary/value.csv"
    beeswarm_summary_key = f"{base_key}/beeswarm_summary/value.csv"
    try:
        bar_csv_exists = await s3_service.file_exists(
            filename=bar_summary_key, session_id=session_id
        )
        beeswarm_csv_exists = await s3_service.file_exists(
            filename=beeswarm_summary_key, session_id=session_id
        )
        if bar_csv_exists and beeswarm_csv_exists:
            bar_summary_data = await s3_service.read(
                f"uploads/{session_id}/{bar_summary_key}"
            )
            beeswarm_summary_data = await s3_service.read(
                f"uploads/{session_id}/{beeswarm_summary_key}"
            )
            bar_summary_df = pd.read_csv(io.BytesIO(bar_summary_data))
            beeswarm_summary_df = pd.read_csv(io.BytesIO(beeswarm_summary_data))
            return {
                "bar_summary": bar_summary_df.to_dict(orient="records"),
                "beeswarm_summary": beeswarm_summary_df.to_dict(orient="records"),
            }
    except Exception:
        # If reading from S3 fails, proceed to recompute.
        pass

    try:
        # Retrieve file data from S3 using session data.
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session expired or not found",
            )
        file_data = await s3_service.read(session_data)

        # Define a helper function for heavy (CPU-bound) computations.
        def compute_xai_summary(file_data: bytes, attack_type: str):
            # Load dataset and reset the index.
            file_like_object = io.BytesIO(file_data)
            df = pd.read_csv(
                file_like_object, engine="pyarrow", dtype_backend="pyarrow"
            )
            df.reset_index(drop=True, inplace=True)

            # Load model artifacts.
            model = xgb.Booster()
            model_path = os.path.join("model", "xgb_booster.model")
            scaler_path = os.path.join("model", "scaler.pkl")
            label_enc_path = os.path.join("model", "label_encoder.pkl")
            model.load_model(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(label_enc_path)

            explainer = shap.TreeExplainer(model)

            # Prepare features & compute SHAP.
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
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

            shap_values = explainer.shap_values(X_scaled_df)

            try:
                attack_class_index = label_encoder.transform([attack_type])[0]
            except ValueError:
                raise ValueError(f"'{attack_type}' not recognized in model classes.")

            shap_values_attack_class = shap_values[..., attack_class_index]

            # Compute bar summary.
            mean_abs_shap_values = np.mean(np.abs(shap_values_attack_class), axis=0)
            sorted_idx = np.argsort(mean_abs_shap_values)
            sorted_features_desc = [feature_cols[i] for i in sorted_idx[::-1]]
            sorted_importances_desc = mean_abs_shap_values[sorted_idx[::-1]]
            bar_summary_df = pd.DataFrame(
                {
                    "feature": sorted_features_desc,
                    "mean_abs_shap": sorted_importances_desc,
                }
            )

            # Compute beeswarm summary.
            shap_df = pd.DataFrame(
                shap_values_attack_class, columns=X_scaled_df.columns
            )
            shap_df["index"] = shap_df.index
            shap_df = shap_df.melt(
                id_vars=["index"], var_name="feature", value_name="shap_value"
            )
            # Melt the scaled feature values (for color mapping).
            feature_vals = X_scaled_df.melt(ignore_index=False).reset_index()
            feature_vals.columns = ["index", "feature", "feature_value"]
            melted_df = shap_df.merge(feature_vals, on=["index", "feature"])
            # Melt the original feature values (for hover display).
            orig_vals = X.melt(ignore_index=False).reset_index()
            orig_vals.columns = ["index", "feature", "original_feature_value"]
            melted_df = melted_df.merge(orig_vals, on=["index", "feature"])
            # Compute normalized feature values.
            melted_df["normalized_feature_value"] = melted_df.groupby("feature")[
                "feature_value"
            ].transform(normalize_feature)
            # Order features by average absolute SHAP value.
            feature_order = (
                melted_df.groupby("feature")["shap_value"]
                .apply(lambda x: np.mean(np.abs(x)))
                .sort_values(ascending=True)
                .index.tolist()
            )
            feature_mapping = {feature: i for i, feature in enumerate(feature_order)}
            # Compute per-feature jitter.
            np.random.seed(42)
            melted_df["jitter_offset"] = melted_df.groupby("feature")[
                "shap_value"
            ].transform(
                lambda s: pd.Series(
                    compute_beeswarm_jitter(s.values, row_height=0.4), index=s.index
                )
            )
            melted_df["y_jitter"] = (
                melted_df["feature"].map(feature_mapping) + melted_df["jitter_offset"]
            )
            melted_df.drop(
                columns=["feature_value", "jitter_offset", "index"], inplace=True
            )

            beeswarm_summary_df = melted_df.copy()
            # --- Sampling Step: Limit to 1000 records, with equal sampling per feature.
            if len(beeswarm_summary_df) > 1000:
                unique_features = beeswarm_summary_df["feature"].unique()
                records_per_feature = 1000 // len(unique_features)
                sampled_dfs = []
                for feat in unique_features:
                    feat_df = beeswarm_summary_df[
                        beeswarm_summary_df["feature"] == feat
                    ]
                    if len(feat_df) > records_per_feature:
                        feat_sample = feat_df.sample(
                            n=records_per_feature, random_state=42
                        )
                    else:
                        feat_sample = feat_df
                    sampled_dfs.append(feat_sample)
                beeswarm_summary_df = pd.concat(sampled_dfs).reset_index(drop=True)
            # --- End Sampling Step

            return bar_summary_df, beeswarm_summary_df

        # Offload the heavy computation to a worker thread.
        bar_summary_df, beeswarm_summary_df = await asyncio.to_thread(
            compute_xai_summary, file_data, attack_type
        )

        # Upload the bar summary CSV to S3.
        bar_summary_csv_buffer = io.BytesIO()
        bar_summary_df.to_csv(bar_summary_csv_buffer, index=False)
        bar_summary_csv_buffer.seek(0)
        await s3_service.upload(
            file=UploadFile(file=bar_summary_csv_buffer),
            file_path=bar_summary_key,
            session_id=session_id,
        )

        # Upload the beeswarm summary CSV to S3.
        beeswarm_summary_csv_buffer = io.BytesIO()
        beeswarm_summary_df.to_csv(beeswarm_summary_csv_buffer, index=False)
        beeswarm_summary_csv_buffer.seek(0)
        await s3_service.upload(
            file=UploadFile(file=beeswarm_summary_csv_buffer),
            file_path=beeswarm_summary_key,
            session_id=session_id,
        )

        return {
            "bar_summary": bar_summary_df.to_dict(orient="records"),
            "beeswarm_summary": beeswarm_summary_df.to_dict(orient="records"),
        }

    except Exception as e:
        print("ERROR", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to retrieve or process data.",
        )


@router.get("/summary/bar/explanation")
async def get_attack_detection_xai_summary_bar_explanation(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
    gemini_service: GeminiService = Depends(GeminiService),
):
    """
    Retrieve or generate an explanation for the SHAP summary bar CSV for a given attack_type.
    This explanation is generated using the Gemini API based on the bar summary data.

    Steps:
      1. Validate the session.
      2. Check if the bar-summary CSV file exists in S3.
      3. If a Gemini explanation file already exists, return its content.
      4. Otherwise, read the CSV data from S3, generate the explanation using the Gemini service,
         upload the explanation to S3, and return it.
    """
    # Validate session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing")

    print("session_id", session_id)

    base_key = f"xai/{attack_type}/summary"
    bar_summary_key = f"{base_key}/bar_summary/value.csv"
    gemini_explanation_key = f"{base_key}/bar_summary/explanation.text"

    # Check if the bar-summary CSV exists in S3
    try:
        csv_exists = await s3_service.file_exists(
            filename=bar_summary_key, session_id=session_id
        )
    except Exception as exc:
        print("Error checking S3 for bar-summary CSV file:", exc)
        raise HTTPException(
            status_code=500, detail="Error checking S3 for bar-summary CSV file"
        )

    print("csv_exists", csv_exists)
    if not csv_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Bar-summary CSV not found for attack type '{attack_type}'.",
        )

    # Check if the Gemini explanation already exists in S3
    try:
        explanation_exists = await s3_service.file_exists(
            filename=gemini_explanation_key, session_id=session_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Error checking S3 for explanation file"
        )

    if explanation_exists:
        try:
            explanation_data = await s3_service.read(
                f"uploads/{session_id}/{gemini_explanation_key}"
            )
            explanation_text = explanation_data.decode("utf-8")
            return {"explanation": explanation_text}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail="Error reading explanation file from S3"
            )

    # Read the bar-summary CSV file from S3
    try:
        print("Reading bar-summary CSV file from S3")
        bar_summary_data = await s3_service.read(
            f"uploads/{session_id}/{bar_summary_key}"
        )
        bar_summary_csv = bar_summary_data.decode("utf-8")
        print("bar_summary_csv\n", bar_summary_csv)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Error reading bar-summary CSV file from S3"
        )

    # Generate the explanation using Gemini
    try:
        explanation = gemini_service.generate_shap_summary_bar_explanation(
            input_shap_summary_bar_data=bar_summary_csv,
            attackType=attack_type,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Error generating explanation from Gemini API"
        )

    # Upload the generated explanation to S3 for future reuse
    try:
        upload_explanation_file = UploadFile(
            filename="explanation.text",
            file=io.BytesIO(explanation.encode("utf-8")),
        )
        await s3_service.upload(
            file=upload_explanation_file,
            file_path=gemini_explanation_key,
            session_id=session_id,
        )
    except Exception as exc:
        # Log the exception, but do not block the response if saving fails.
        print("Warning: Failed to save explanation file to S3:", exc)

    return {"explanation": explanation}


@router.get("/summary/beeswarm/explanation")
async def get_attack_detection_xai_summary_beeswarm_explanation(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
    gemini_service: GeminiService = Depends(GeminiService),
):
    """
    Retrieve or generate an explanation for the SHAP beeswarm summary CSV for a given attack_type.
    This explanation is generated using the Gemini API based on the beeswarm summary data.

    Steps:
      1. Validate the session ID.
      2. Check if the beeswarm-summary CSV file exists in S3.
      3. If a Gemini explanation file already exists, return its content.
      4. Otherwise, read the CSV data from S3, filter it to include only the top 2 objects
         per unique feature (based on the largest absolute SHAP value), generate the explanation
         using the Gemini service, upload the explanation to S3, and return it.
    """
    # Validate session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing")

    # Define S3 file keys
    base_key = f"xai/{attack_type}/summary"
    beeswarm_summary_key = f"{base_key}/beeswarm_summary/value.csv"
    gemini_explanation_key = f"{base_key}/beeswarm_summary/explanation.text"

    # Check if the beeswarm-summary CSV exists in S3
    try:
        csv_exists = await s3_service.file_exists(
            filename=beeswarm_summary_key, session_id=session_id
        )
    except Exception as exc:
        print("Error checking S3 for beeswarm-summary CSV file:", exc)
        raise HTTPException(
            status_code=500, detail="Error checking S3 for beeswarm-summary CSV file"
        )

    if not csv_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Beeswarm-summary CSV not found for attack type '{attack_type}'.",
        )

    # Check if the Gemini explanation already exists in S3
    try:
        explanation_exists = await s3_service.file_exists(
            filename=gemini_explanation_key, session_id=session_id
        )
    except Exception as exc:
        print("Error checking S3 for existing explanation file:", exc)
        raise HTTPException(
            status_code=500, detail="Error checking S3 for existing explanation file"
        )

    # if explanation_exists:
    #     try:
    #         explanation_data = await s3_service.read(
    #             f"uploads/{session_id}/{gemini_explanation_key}"
    #         )
    #         explanation_text = explanation_data.decode("utf-8")
    #         return {"explanation": explanation_text}
    #     except Exception as exc:
    #         print("Error reading existing explanation file from S3:", exc)
    #         raise HTTPException(
    #             status_code=500,
    #             detail="Error reading existing explanation file from S3",
    #         )

    # Read the full beeswarm-summary CSV from S3
    try:
        beeswarm_summary_data = await s3_service.read(
            f"uploads/{session_id}/{beeswarm_summary_key}"
        )
        beeswarm_summary_csv = beeswarm_summary_data.decode("utf-8")
    except Exception as exc:
        print("Error reading beeswarm-summary CSV file from S3:", exc)
        raise HTTPException(
            status_code=500, detail="Error reading beeswarm-summary CSV file from S3"
        )

    # Filter the CSV: select only the top 2 rows per unique feature based on largest |shap_value|
    try:
        # Use pandas to parse and filter the data
        from io import StringIO
        import pandas as pd

        df = pd.read_csv(StringIO(beeswarm_summary_csv))
        # Group by "feature" and take the top 2 rows sorted by absolute shap_value
        filtered_df = df.groupby("feature", group_keys=False).apply(
            lambda x: x.reindex(
                x["shap_value"].abs().sort_values(ascending=False).index
            ).head(2)
        )

        # Select only the columns needed for the explanation
        filtered_df = filtered_df[["feature", "original_feature_value", "shap_value"]]
        print("filtered_df\n", filtered_df)
    except Exception as exc:
        print("Error filtering beeswarm-summary CSV data:", exc)
        raise HTTPException(
            status_code=500, detail="Error processing beeswarm-summary CSV data"
        )

    # Generate the explanation using Gemini with the filtered CSV data
    try:
        print("Generating explanation from Gemini API")
        explanation = gemini_service.generate_shap_summary_beeswarm_explanation(
            input_shap_summary_beeswarm_data=filtered_df.to_string(index=False),
            attackType=attack_type,
        )
    except Exception as exc:
        print("Error generating explanation from Gemini API:", exc)
        raise HTTPException(
            status_code=500, detail="Error generating explanation from Gemini API"
        )

    # Upload the generated explanation to S3 for future reuse
    try:
        explanation_bytes = explanation.encode("utf-8")
        upload_explanation_file = UploadFile(
            filename="explanation.text", file=io.BytesIO(explanation_bytes)
        )
        await s3_service.upload(
            file=upload_explanation_file,
            file_path=gemini_explanation_key,
            session_id=session_id,
        )
    except Exception as exc:
        print("Warning: Failed to save beeswarm explanation file to S3:", exc)
        # Do not block returning the explanation if saving fails

    return {"explanation": explanation}
