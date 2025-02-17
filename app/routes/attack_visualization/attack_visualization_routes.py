from datetime import timedelta

from app.schemas.attack_visualization import GetTimeSeriesAttackDataResponse
from fastapi import APIRouter, Depends, HTTPException, Cookie, Query
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
    partition_index: int = Query(0, ge=0),
    s3_service: S3 = Depends(S3),
):
    """
    Return a JSON payload of time-series data, partitioned by 1-hour gaps.
    By default, returns data for partition_index=0 (the first partition).

    The response includes:
      - data: {
          timestamps: [...],
          values: [...],
          attackMarkPoint: [[ts, val], ...],
          otherAttackMarkPoint: [[ts, val], ...],
          feature: <dynamic feature column>
        },
      - highlight: [ [ {name, xAxis}, {xAxis} ], ... ],
      - partitions: [ {start: <ISO>, end: <ISO>}, ... ]
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing")

    # Retrieve the S3 key from Redis
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
        attackTypeFeatureMap = {
            "DDoS": "Bwd Packet Length Mean",
            "FTP-Patator": "Total Length of Fwd Packet",
        }
        feature_name = attackTypeFeatureMap.get(attack_type, "Flow Bytes/s")

        # Parse Timestamp as datetime (assuming ms)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)

        # Check if chosen feature exists
        if feature_name not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{feature_name}' not found in CSV columns."
            )

        # Convert chosen feature to numeric and drop invalid rows
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
        df.dropna(subset=[feature_name], inplace=True)

        # -----------------------------------------------------------------------
        # 3. Define numeric flags for attack classification
        # -----------------------------------------------------------------------
        df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
        df["is_other_attack"] = np.where(
            (df["Label"] != "BENIGN") & (df["Label"] != attack_type),
            1,
            0
        )

        # -----------------------------------------------------------------------
        # 4. Resample by 1-second bins
        # -----------------------------------------------------------------------
        df.set_index("Timestamp", inplace=True)
        grouped = df.resample("1s").agg(
            FeatureMean=(feature_name, "mean"),
            AttackTypeCount=("is_attack_type", "sum"),
            OtherAttackCount=("is_other_attack", "sum"),
            RowCount=(feature_name, "count"),
        )
        grouped = grouped[grouped["RowCount"] > 0]

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
        # 5. Partition the time-series based on gaps > 1 hour
        # -----------------------------------------------------------------------
        partitions_info = []  # List of tuples: (start_idx, end_idx, start_ts, end_ts)
        if len(grouped) == 0:
            return {
                "data": {
                    "timestamps": [],
                    "values": [],
                    "attackMarkPoint": [],
                    "otherAttackMarkPoint": [],
                    "feature": feature_name,
                },
                "highlight": [],
                "partitions": []
            }

        current_start_idx = 0
        prev_timestamp = grouped.loc[0, "Timestamp"]

        for i in range(1, len(grouped)):
            current_timestamp = grouped.loc[i, "Timestamp"]
            if current_timestamp - prev_timestamp > timedelta(hours=1):
                start_ts = grouped.loc[current_start_idx, "Timestamp"]
                end_ts = grouped.loc[i - 1, "Timestamp"]
                partitions_info.append((current_start_idx, i - 1, start_ts, end_ts))
                current_start_idx = i
            prev_timestamp = current_timestamp

        start_ts = grouped.loc[current_start_idx, "Timestamp"]
        end_ts = grouped.loc[len(grouped) - 1, "Timestamp"]
        partitions_info.append((current_start_idx, len(grouped) - 1, start_ts, end_ts))

        partitions_boundary = [
            {"start": st.isoformat(), "end": et.isoformat()}
            for (_, _, st, et) in partitions_info
        ]

        # -----------------------------------------------------------------------
        # 6. Validate partition_index
        # -----------------------------------------------------------------------
        if partition_index >= len(partitions_info):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid partition_index {partition_index}. Only {len(partitions_info)} partition(s) available."
            )

        # Slice the DataFrame to the requested partition
        part_start_i, part_end_i, part_st, part_et = partitions_info[partition_index]
        df_part = grouped.iloc[part_start_i : part_end_i + 1].copy()

        # -----------------------------------------------------------------------
        # 7. Build arrays for timestamps, values, and mark points for this partition
        # -----------------------------------------------------------------------
        timestamps = []
        values = []
        attackMarkPoint = []
        otherAttackMarkPoint = []

        for _, row in df_part.iterrows():
            ts = row["Timestamp"]
            val = row["FeatureMean"] if not pd.isna(row["FeatureMean"]) else 0.0
            rounded_val = round(float(val), 3)
            timestamps.append(ts.isoformat())
            values.append(rounded_val)

            if row["attack"] == "match":
                attackMarkPoint.append([ts.isoformat(), rounded_val])
            elif row["attack"] == "other":
                otherAttackMarkPoint.append([ts.isoformat(), rounded_val])

        # -----------------------------------------------------------------------
        # 8. Build highlight intervals for this partition
        # -----------------------------------------------------------------------
        df_part["attack_name"] = df_part["attack"].apply(
            lambda x: attack_type if x == "match" else ("otherAttack" if x == "other" else None)
        )
        highlight = []
        current_attack = None
        start_ts_p = None
        prev_ts_p = None

        for _, row in df_part.iterrows():
            row_attack = row["attack_name"]
            ts = row["Timestamp"]

            if row_attack is None:
                if current_attack is not None and start_ts_p is not None:
                    end_ts_p = prev_ts_p if prev_ts_p is not None else ts
                    highlight.append([
                        {"name": current_attack, "xAxis": start_ts_p.isoformat()},
                        {"xAxis": end_ts_p.isoformat()}
                    ])
                    current_attack = None
                    start_ts_p = None
            else:
                if current_attack is None:
                    current_attack = row_attack
                    start_ts_p = ts
                elif row_attack != current_attack:
                    if start_ts_p is not None:
                        end_ts_p = prev_ts_p if prev_ts_p is not None else ts
                        highlight.append([
                            {"name": current_attack, "xAxis": start_ts_p.isoformat()},
                            {"xAxis": end_ts_p.isoformat()}
                        ])
                    current_attack = row_attack
                    start_ts_p = ts
            prev_ts_p = ts

        if current_attack is not None and start_ts_p is not None:
            highlight.append([
                {"name": current_attack, "xAxis": start_ts_p.isoformat()},
                {"xAxis": prev_ts_p.isoformat()}
            ])

        # -----------------------------------------------------------------------
        # 9. Build final response object
        # -----------------------------------------------------------------------
        response_json = {
            "data": {
                "timestamps": timestamps,
                "values": values,
                "attackMarkPoint": attackMarkPoint,
                "otherAttackMarkPoint": otherAttackMarkPoint,
                "feature": feature_name,
            },
            "highlight": highlight,
            "partitions": partitions_boundary,
        }

        return response_json

    except Exception as e:
        # If the caught exception is an HTTPException, re-raise it.
        if isinstance(e, HTTPException):
            raise e
        print("Error in get_time_series_attack_data:", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve time-series data."
        )
