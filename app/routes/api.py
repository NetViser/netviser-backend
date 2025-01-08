import uuid
from fastapi import APIRouter, File, Response, UploadFile
from app.configs.config import get_settings
import xgboost as xgb
import pandas as pd
import os

router = APIRouter(prefix="/api", tags=["api"])
settings = get_settings()


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


@router.post("/api/session")
def create_session(response: Response):
    session_id = str(uuid.uuid4())
    # new_session = SessionModel(session_id=session_id)
    # db.add(new_session)
    # db.commit()
    response.set_cookie(
        key="session_id", value=session_id, httponly=True, max_age=86400
    )  # 1 day
    return {"session_id": session_id}
