import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


feature_columns = [
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


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by handling missing values and extracting metadata."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index
    irrelevant_columns = ["Flow ID", "Attempted Category", "Hour", "Day"]
    df_processed = df.drop(columns=irrelevant_columns, axis=1, errors="ignore")
    return df_processed


def predict_df(X, model, scaler, label_encoder):
    """Perform model prediction and return labeled outputs."""
    X_scaled = scaler.transform(X)
    dmatrix = xgb.DMatrix(X_scaled)
    predictions = model.predict(dmatrix)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    return predicted_labels


def get_model_artifacts():
    """Load model artifacts from disk."""
    model_file = os.path.join("model", "xgb_booster.model")
    model = xgb.Booster()
    model.load_model(model_file)

    # Load the scaler and label encoder
    scaler_file = os.path.join("model", "scaler.pkl")
    scaler = joblib.load(scaler_file)
    label_encoder_file = os.path.join("model", "label_encoder.pkl")
    label_encoder = joblib.load(label_encoder_file)
    return model, scaler, label_encoder