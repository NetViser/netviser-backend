from app.schemas.attack_visualization import GetTimeSeriesAttackDataResponse

from fastapi import APIRouter, Depends, HTTPException, Cookie
from typing import Optional
import pandas as pd
import numpy as np
import io

from app.services.s3_service import S3
from app.services.redis_service import RedisClient
from app.configs.config import get_settings

router = APIRouter(prefix="/api/attack-detection/visualization", tags=["visualization"])
settings = get_settings()
redis_client = RedisClient()

@router.get("/attack-time-series", response_model=GetTimeSeriesAttackDataResponse)
async def get_time_series_attack_data(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Return a JSON payload of time-series data, including arrays for timestamps, values,
    plus separate markPoint arrays for rows labeled "match" vs. "other", and a
    'highlight' list of intervals.

    The output format:
    {
      "data": {
        "timestamps": [...],
        "values": [...],
        "attackMarkPoint": [[ts, val], [ts, val], ...],
        "otherAttackMarkPoint": [[ts, val], ...],
        "feature": "<dynamic feature column>"
      },
      "highlight": [
        [
          {"name": "DDoS", "xAxis": "..."},
          {"xAxis": "..."}
        ],
        [
          {"name": "otherAttack", "xAxis": "..."},
          {"xAxis": "..."}
        ]
      ]
    }
    """

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing")

    # Get the S3 key from Redis
    s3_key = redis_client.get_session_data(session_id)
    if not s3_key:
        raise HTTPException(status_code=400, detail="Session Expired or not found")

    try:
        # -----------------------------------------------------------------------
        # 1. Read Data from S3
        # -----------------------------------------------------------------------
        file_data = await s3_service.read(s3_key)
        file_like_object = io.BytesIO(file_data)

        df = pd.read_csv(file_like_object, engine="pyarrow", dtype_backend="pyarrow")
        print("Original data shape:", df.shape)

        # -----------------------------------------------------------------------
        # 2. Decide which feature column to chart, based on attack_type
        # -----------------------------------------------------------------------
        # e.g. you can map DDoS -> 'Bwd Packet Length Mean', FTP-Patator -> 'Total Length of Fwd Packet'
        attackTypeFeatureMap = {
            "DDoS": "Bwd Packet Length Mean",
            "FTP-Patator": "Total Length of Fwd Packet",
        }
        feature_name = attackTypeFeatureMap.get(attack_type, "Flow Bytes/s")

        # Parse Timestamp as datetime (assuming ms)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)

        # -----------------------------------------------------------------------
        # 3. Convert the chosen feature column to numeric, drop invalid
        # -----------------------------------------------------------------------
        if feature_name not in df.columns:
            # If the feature doesn't exist in this dataset, fallback or raise error
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{feature_name}' not found in CSV columns."
            )

        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
        df.dropna(subset=[feature_name], inplace=True)

        # -----------------------------------------------------------------------
        # 4. Identify "is_attack_type" vs "is_other_attack"
        # -----------------------------------------------------------------------
        df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
        df["is_other_attack"] = np.where(
            (df["Label"] != "BENIGN") & (df["Label"] != attack_type),
            1,
            0
        )

        # -----------------------------------------------------------------------
        # 5. Resample by 1-second bins
        # -----------------------------------------------------------------------
        df.set_index("Timestamp", inplace=True)
        grouped = df.resample("1s").agg(
            FeatureMean=(feature_name, "mean"),
            AttackTypeCount=("is_attack_type", "sum"),
            OtherAttackCount=("is_other_attack", "sum"),
            RowCount=(feature_name, "count"),
        )

        # Keep only bins with >= 1 row
        grouped = grouped[grouped["RowCount"] > 0]

        # Decide "match"/"other"/"none" for each bin
        def label_bin(row):
            if row["AttackTypeCount"] > 0:
                return "match"
            elif row["OtherAttackCount"] > 0:
                return "other"
            return "none"

        grouped["attack"] = grouped.apply(label_bin, axis=1)

        grouped.index.name = "Timestamp"
        grouped.reset_index(inplace=True)
        grouped.sort_values("Timestamp", inplace=True)

        # -----------------------------------------------------------------------
        # 6. Build arrays for timestamps, values, plus MarkPoints
        # -----------------------------------------------------------------------
        timestamps = []
        values = []
        attackMarkPoint = []
        otherAttackMarkPoint = []

        for _, row in grouped.iterrows():
            ts = row["Timestamp"]
            val = row["FeatureMean"] if not pd.isna(row["FeatureMean"]) else 0.0

            # Store main arrays
            timestamps.append(ts.isoformat())
            values.append(float(val))

            if row["attack"] == "match":
                attackMarkPoint.append([ts.isoformat(), float(val)])
            elif row["attack"] == "other":
                otherAttackMarkPoint.append([ts.isoformat(), float(val)])

        # -----------------------------------------------------------------------
        # 7. Build highlight intervals
        # -----------------------------------------------------------------------
        # We'll define 'attack_name' => user-specified for "match",
        # "otherAttack" for "other", else None
        grouped["attack_name"] = grouped["attack"].apply(
            lambda x: attack_type if x == "match" else ("otherAttack" if x == "other" else None)
        )

        highlight = []
        current_attack = None
        start_ts = None
        prev_ts = None

        for _, row in grouped.iterrows():
            row_attack = row["attack_name"]
            ts = row["Timestamp"]

            if row_attack is None:
                # If we currently have an open run, close it
                if current_attack is not None and start_ts is not None:
                    end_ts = prev_ts if prev_ts else ts
                    highlight.append([
                        {"name": current_attack, "xAxis": start_ts.isoformat()},
                        {"xAxis": end_ts.isoformat()}
                    ])
                    current_attack = None
                    start_ts = None
            else:
                # row_attack is the user-specified attack_type or "otherAttack"
                if current_attack is None:
                    # Start new run
                    current_attack = row_attack
                    start_ts = ts
                elif row_attack != current_attack:
                    # Attack changed => close old run
                    if start_ts is not None:
                        end_ts = prev_ts if prev_ts else ts
                        highlight.append([
                            {"name": current_attack, "xAxis": start_ts.isoformat()},
                            {"xAxis": end_ts.isoformat()}
                        ])
                    current_attack = row_attack
                    start_ts = ts

            prev_ts = ts

        # If there's an open run at the end
        if current_attack is not None and start_ts is not None:
            highlight.append([
                {"name": current_attack, "xAxis": start_ts.isoformat()},
                {"xAxis": prev_ts.isoformat()}
            ])

        # -----------------------------------------------------------------------
        # 8. Build final response object
        # -----------------------------------------------------------------------
        response_json = {
            "data": {
                "timestamps": timestamps,
                "values": values,
                "attackMarkPoint": attackMarkPoint,
                "otherAttackMarkPoint": otherAttackMarkPoint,
                "feature": feature_name,  # <--- use dynamic feature here
            },
            "highlight": highlight,
        }

        return response_json

    except Exception as e:
        print("Error in get_time_series_attack_data:", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve time-series data."
        )
