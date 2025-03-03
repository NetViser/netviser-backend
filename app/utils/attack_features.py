import pandas as pd
import numpy as np


def portscan_feature_extraction(
    df: pd.DataFrame, attack_type: str
) -> tuple[pd.DataFrame, str]:
    """Extracts the number of unique dstPort values per second for Portscan."""
    if "Dst Port" not in df.columns:
        raise ValueError("Column 'Dst Port' not found in CSV for Portscan analysis.")

    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )

    df.set_index("Timestamp", inplace=True)
    grouped = df.resample("1s").agg(
        dstPortCount=("Dst Port", "nunique"),  # Count unique dstPort values
        AttackTypeCount=("is_attack_type", "sum"),
        OtherAttackCount=("is_other_attack", "sum"),
        RowCount=("is_attack_type", "count"),
    )
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "dstPortCount"


def ddos_feature_extraction(
    df: pd.DataFrame, attack_type: str
) -> tuple[pd.DataFrame, str]:
    """Extracts the number of forward packets per second for DDoS."""
    required_field = "Total Fwd Packet"
    if required_field not in df.columns:
        raise ValueError(f"Missing required column for DDoS analysis: {required_field}")

    # Convert relevant field to numeric and handle NaNs
    df["Total Fwd Packet"] = pd.to_numeric(df["Total Fwd Packet"], errors="coerce")
    df.dropna(subset=[required_field], inplace=True)

    # Define attack classification flags
    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )

    # Resample by 1-second intervals
    df.set_index("Timestamp", inplace=True)
    grouped = df.resample("1s").agg(
        PacketCount=(
            "Total Fwd Packet",
            "sum",
        ),  # Total number of forward packets per second
        AttackTypeCount=("is_attack_type", "sum"),
        OtherAttackCount=("is_other_attack", "sum"),
        RowCount=("Total Fwd Packet", "count"),
    )

    # Filter out rows with no data
    grouped = grouped[grouped["RowCount"] > 0]

    return grouped, "PacketCount"


def ftp_patator_feature_extraction(
    df: pd.DataFrame, attack_type: str
) -> tuple[pd.DataFrame, str]:
    """Extracts the mean of 'Total Length of Fwd Packet' per second for FTP-Patator."""
    feature_name = "Total Length of Fwd Packet"
    if feature_name not in df.columns:
        raise ValueError(
            f"Column '{feature_name}' not found in CSV for FTP-Patator analysis."
        )

    df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
    df.dropna(subset=[feature_name], inplace=True)

    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )

    df.set_index("Timestamp", inplace=True)
    grouped = df.resample("1s").agg(
        FeatureMean=(feature_name, "mean"),
        AttackTypeCount=("is_attack_type", "sum"),
        OtherAttackCount=("is_other_attack", "sum"),
        RowCount=(feature_name, "count"),
    )
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "FeatureMean"


def default_feature_extraction(
    df: pd.DataFrame, attack_type: str
) -> tuple[pd.DataFrame, str]:
    """Default feature extraction using 'Flow Bytes/s' for unknown attack types."""
    feature_name = "Flow Bytes/s"
    if feature_name not in df.columns:
        raise ValueError(
            f"Column '{feature_name}' not found in CSV for default analysis."
        )

    df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
    df.dropna(subset=[feature_name], inplace=True)

    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )

    df.set_index("Timestamp", inplace=True)
    grouped = df.resample("1s").agg(
        FeatureMean=(feature_name, "mean"),
        AttackTypeCount=("is_attack_type", "sum"),
        OtherAttackCount=("is_other_attack", "sum"),
        RowCount=(feature_name, "count"),
    )
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "FeatureMean"
