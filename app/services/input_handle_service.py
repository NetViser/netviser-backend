from fastapi import Response
import pandas as pd


async def preprocess(file_like_object):
    data_frame = pd.read_csv(
        file_like_object, engine="pyarrow", dtype_backend="pyarrow"
    )

    # Ensure 'Label' and 'Timestamp' columns exist
    if "Label" not in data_frame.columns:
        return Response(status_code=400, content="Label column missing in the dataset")

    if "Timestamp" not in data_frame.columns:
        return Response(
            status_code=400, content="Timestamp column missing in the dataset"
        )

    # Convert Timestamp to datetime and sort
    data_frame["Timestamp"] = pd.to_datetime(data_frame["Timestamp"])
    data_frame.sort_values(by="Timestamp", inplace=True)

    # 1) Resample Flow Bytes/s to 1-second intervals using mean
    #    - set index as Timestamp
    data_frame.set_index("Timestamp", inplace=True)

    return data_frame


