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


@router.get("/individual/explaination")
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
        explanation = gemini_service.generate_shap_force_plot_explanation(shap_text)
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

    # Check if bar-summary CSV exists in S3
    base_key = f"xai/{attack_type}/summary"
    bar_summary_key = f"{base_key}/bar_summary/value.csv"
    beeswarm_summary_key = f"{base_key}/beeswarm_summary/value.csv"
    # try:
    #     csv_exists = await s3_service.file_exists(
    #         filename=bar_summary_key, session_id=session_id
    #     )
    #     if csv_exists:
    #         bar_summary_data = await s3_service.read(
    #             f"uploads/{session_id}/{bar_summary_key}"
    #         )

    #         bar_summary_df = pd.read_csv(io.BytesIO(bar_summary_data))
    #         print("bar_summary_df\n", bar_summary_df)
    #         return {"bar_summary": bar_summary_df.to_dict(orient="records")}
    # except Exception:
    #.    pass  # Not a hard failure, continue to generate the CSV

    # Compute the SHAP-based summary
    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session Expired or not found",
            )

        # Load dataset from S3
        file_data = await s3_service.read(session_data)
        file_like_object = io.BytesIO(file_data)
        df = pd.read_csv(file_like_object, engine="pyarrow", dtype_backend="pyarrow")
        df.reset_index(drop=True, inplace=True)

        # Load model artifacts
        model = xgb.Booster()
        model_path = os.path.join("model", "xgb_booster.model")
        scaler_path = os.path.join("model", "scaler.pkl")
        label_enc_path = os.path.join("model", "label_encoder.pkl")

        try:
            model.load_model(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(label_enc_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model or scaler file missing on the server.",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model artifacts.",
            )

        explainer = shap.TreeExplainer(model)

        # Prepare features & compute SHAP
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
        print("shap_values\n", shap_values[0])

        # Extract SHAP values for the requested attack_type
        try:
            attack_class_index = label_encoder.transform([attack_type])[0]
            print("attack_class_index", attack_class_index)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{attack_type}' not recognized in model classes.",
            )

        shap_values_attack_class = shap_values[..., attack_class_index]
        print("after shap_values_attack_class")

        # Compute csv data for bar summary
        mean_abs_shap_values = np.mean(np.abs(shap_values_attack_class), axis=0)
        print("after mean_abs_shap_values")
        sorted_idx = np.argsort(mean_abs_shap_values)
        print("after sorted_idx")
        sorted_features_desc = [feature_cols[i] for i in sorted_idx[::-1]]
        print("after sorted_features_desc")
        sorted_importances_desc = mean_abs_shap_values[sorted_idx[::-1]]

        # Create DataFrame & upload to S3
        bar_summary_df = pd.DataFrame(
            {"feature": sorted_features_desc, "mean_abs_shap": sorted_importances_desc}
        )
        print("bar_summary_df\n", bar_summary_df)
        bar_summary_csv_buffer = io.BytesIO()
        bar_summary_df.to_csv(bar_summary_csv_buffer, index=False)
        bar_summary_csv_buffer.seek(0)

        await s3_service.upload(
            file=UploadFile(file=bar_summary_csv_buffer),
            file_path=bar_summary_key,
            session_id=session_id,
        )

        # Compute csv data for beeswarm summary
        shap_df = pd.DataFrame(shap_values_attack_class, columns=X_scaled_df.columns)
        shap_df["index"] = shap_df.index
        shap_df = shap_df.melt(
            id_vars=["index"], var_name="feature", value_name="shap_value"
        )
        # Melt the continuous feature values from the scaled data (for color mapping)
        feature_vals = X_scaled_df.melt(ignore_index=False).reset_index()
        feature_vals.columns = ["index", "feature", "feature_value"]
        # Merge SHAP values with the scaled feature values
        melted_df = shap_df.merge(feature_vals, on=["index", "feature"])
        # Also melt the original (unscaled) feature values for hover display
        orig_vals = X.melt(ignore_index=False).reset_index()
        orig_vals.columns = ["index", "feature", "original_feature_value"]
        melted_df = melted_df.merge(orig_vals, on=["index", "feature"])
        # Compute percentile-based normalized value per feature using the scaled "Feature Value"
        melted_df["normalized_feature_value"] = melted_df.groupby("feature")[
            "feature_value"
        ].transform(normalize_feature)
        # -------------------------
        # Prepare for Beeswarm Plotting
        # -------------------------
        # Order features (for example, by average absolute SHAP value)
        feature_order = (
            melted_df.groupby("feature")["shap_value"]
            .apply(lambda x: np.mean(np.abs(x)))
            .sort_values(ascending=True)
            .index.tolist()
        )
        # Map each feature to a numeric y-value for plotting
        feature_mapping = {feature: i for i, feature in enumerate(feature_order)}
        # After computing feature_mapping (which maps each feature to a base numeric y value),
        # replace the uniform jitter with a jitter computed per feature based on the SHAP value distribution.
        np.random.seed(42)
        # Instead of:
        # melted_df["y_jitter"] = melted_df["Feature"].map(feature_mapping) + np.random.uniform(-0.3, 0.3, size=len(melted_df))
        # we compute a per-feature jitter:
        # First compute the jitter offset per feature:
        melted_df["jitter_offset"] = melted_df.groupby("feature")[
            "shap_value"
        ].transform(
            lambda s: pd.Series(
                compute_beeswarm_jitter(s.values, row_height=0.4), index=s.index
            )
        )

        # Then add the base y value from feature_mapping to the computed offset:
        melted_df["y_jitter"] = (
            melted_df["feature"].map(feature_mapping) + melted_df["jitter_offset"]
        )
        melted_df.drop(
            columns=["feature_value", "jitter_offset", "index"], inplace=True
        )

        # Upload the beeswarm summary CSV to S3
        beeswarm_summary_df = melted_df.copy()
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
            "beeswarm_summary": {}# beeswarm_summary_df.to_dict(orient="records"),
        }

    except Exception as e:
        print("ERROR", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to retrieve or process data.",
        )
