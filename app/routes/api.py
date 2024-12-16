from fastapi import APIRouter, File, UploadFile
import xgboost as xgb
import pandas as pd
import os

router = APIRouter(prefix="/api", tags=["api"])


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
