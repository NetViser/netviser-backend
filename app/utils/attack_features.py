import pandas as pd
import numpy as np

def add_ports_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds flag columns for ports 21, 22, and 20 based on the 'Dst Port' column.
    """
    if "Dst Port" not in df.columns:
        raise ValueError("Column 'Dst Port' not found in DataFrame.")
    df["is_port21"] = np.where(df["Dst Port"] == 21, 1, 0)
    df["is_port22"] = np.where(df["Dst Port"] == 22, 1, 0)
    df["is_port20"] = np.where(df["Dst Port"] == 20, 1, 0)
    return df

def dst_port_count_per_sec_feature_extraction(
    df: pd.DataFrame, attack_type: str, port_flag: bool = False
) -> tuple[pd.DataFrame, str]:
    """Extracts the number of unique dstPort values per second for Portscan."""
    if "Dst Port" not in df.columns:
        raise ValueError("Column 'Dst Port' not found in CSV for Portscan analysis.")

    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )
    
    # If port_flag is true, add port feature flags.
    if port_flag:
        df = add_ports_flag(df)

    df.set_index("Timestamp", inplace=True)
    # Define the base aggregation dictionary.
    agg_dict = {
        "dstPortCount": ("Dst Port", "nunique"),
        "AttackTypeCount": ("is_attack_type", "sum"),
        "OtherAttackCount": ("is_other_attack", "sum"),
        "RowCount": ("is_attack_type", "count"),
    }
    # If port_flag is true, update the aggregation with port flag aggregations.
    if port_flag:
        port_aggs = {
            "is_port21": ("is_port21", "max"),
            "is_port22": ("is_port22", "max"),
            "is_port20": ("is_port20", "max")
        }
        agg_dict = {**agg_dict, **port_aggs}
        
    grouped = df.resample("1s").agg(**agg_dict)
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "dstPortCount"


def packet_count_per_sec_feature_extraction(
    df: pd.DataFrame, attack_type: str, port_flag: bool = False
) -> tuple[pd.DataFrame, str]:
    """Extracts the sum of 'Total Fwd Packet' per second for attack analysis."""
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
    
    # If port_flag is true, ensure port columns exist and add port features.
    if port_flag:
        if "Dst Port" not in df.columns:
            raise ValueError("Column 'Dst Port' not found for port features.")
        df = add_ports_flag(df)

    df.set_index("Timestamp", inplace=True)
    agg_dict = {
        "PacketCount": ("Total Fwd Packet", "sum"),
        "AttackTypeCount": ("is_attack_type", "sum"),
        "OtherAttackCount": ("is_other_attack", "sum"),
        "RowCount": ("Total Fwd Packet", "count"),
    }
    if port_flag:
        port_aggs = {
            "is_port21": ("is_port21", "max"),
            "is_port22": ("is_port22", "max"),
            "is_port20": ("is_port20", "max")
        }
        agg_dict = {**agg_dict, **port_aggs}
        
    grouped = df.resample("1s").agg(**agg_dict)
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "PacketCount"


def total_length_of_fwd_packet_feature_extraction(
    df: pd.DataFrame, attack_type: str, port_flag: bool = False
) -> tuple[pd.DataFrame, str]:
    """Extracts the mean of 'Total Length of Fwd Packet' per second for attack analysis."""
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
    
    if port_flag:
        if "Dst Port" not in df.columns:
            raise ValueError("Column 'Dst Port' not found for port features.")
        df = add_ports_flag(df)

    df.set_index("Timestamp", inplace=True)
    agg_dict = {
        "FeatureMean": (feature_name, "mean"),
        "AttackTypeCount": ("is_attack_type", "sum"),
        "OtherAttackCount": ("is_other_attack", "sum"),
        "RowCount": (feature_name, "count"),
    }
    if port_flag:
        port_aggs = {
            "is_port21": ("is_port21", "max"),
            "is_port22": ("is_port22", "max"),
            "is_port20": ("is_port20", "max")
        }
        agg_dict = {**agg_dict, **port_aggs}
        
    grouped = df.resample("1s").agg(**agg_dict)
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "FeatureMean"


def default_feature_extraction(
    df: pd.DataFrame, attack_type: str, port_flag: bool = False
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
    
    if port_flag:
        if "Dst Port" not in df.columns:
            raise ValueError("Column 'Dst Port' not found for port features.")
        df = add_ports_flag(df)

    df.set_index("Timestamp", inplace=True)
    agg_dict = {
        "FeatureMean": (feature_name, "mean"),
        "AttackTypeCount": ("is_attack_type", "sum"),
        "OtherAttackCount": ("is_other_attack", "sum"),
        "RowCount": (feature_name, "count"),
    }
    if port_flag:
        port_aggs = {
            "is_port21": ("is_port21", "max"),
            "is_port22": ("is_port22", "max"),
            "is_port20": ("is_port20", "max")
        }
        agg_dict = {**agg_dict, **port_aggs}
        
    grouped = df.resample("1s").agg(**agg_dict)
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "FeatureMean"


def total_tcp_flow_time_feature_extraction(
    df: pd.DataFrame, attack_type: str, port_flag: bool = False
) -> tuple[pd.DataFrame, str]:
    """
    Extracts the maximum of 'Total TCP Flow Time' per second for SSH-Patator analysis.
    If port_flag is true, port features are added; otherwise, port flags are not computed.
    """
    required_feature = "Total TCP Flow Time"
    if required_feature not in df.columns:
        raise ValueError(f"Column '{required_feature}' not found in CSV for SSH-Patator analysis.")
    if "Dst Port" not in df.columns:
        raise ValueError("Column 'Dst Port' not found in CSV for SSH-Patator analysis.")

    df[required_feature] = pd.to_numeric(df[required_feature], errors="coerce")
    df.dropna(subset=[required_feature], inplace=True)
    
    # Add port features only if port_flag is True.
    if port_flag:
        df = add_ports_flag(df)

    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )
    
    df.set_index("Timestamp", inplace=True)
    agg_dict = {
        "FeatureMax": (required_feature, "max"),
        "AttackTypeCount": ("is_attack_type", "sum"),
        "OtherAttackCount": ("is_other_attack", "sum"),
        "RowCount": (required_feature, "count"),
    }
    # If port_flag is true, aggregate all port flags.
    if port_flag:
        port_aggs = {
            "is_port21": ("is_port21", "max"),
            "is_port22": ("is_port22", "max"),
            "is_port20": ("is_port20", "max")
        }
        agg_dict = {**agg_dict, **port_aggs}
        
    grouped = df.resample("1s").agg(**agg_dict)
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "FeatureMax"


def bwd_iat_mean_feature_extraction(
    df: pd.DataFrame, attack_type: str, port_flag: bool = False
) -> tuple[pd.DataFrame, str]:
    """Extracts the mean of 'Bwd IAT Mean' per second for Slowloris DoS analysis."""
    feature_name = "Bwd IAT Mean"
    print("All columns: ", df.columns)
    if feature_name not in df.columns:
        raise ValueError(f"Column '{feature_name}' not found in CSV for Slowloris DoS analysis.")
    
    df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
    df.dropna(subset=[feature_name], inplace=True)
    
    df["is_attack_type"] = np.where(df["Label"] == attack_type, 1, 0)
    df["is_other_attack"] = np.where(
        (df["Label"] != "BENIGN") & (df["Label"] != attack_type), 1, 0
    )
    
    if port_flag:
        if "Dst Port" not in df.columns:
            raise ValueError("Column 'Dst Port' not found for port features.")
        df = add_ports_flag(df)
    
    df.set_index("Timestamp", inplace=True)
    agg_dict = {
        "FeatureMean": (feature_name, "mean"),
        "AttackTypeCount": ("is_attack_type", "sum"),
        "OtherAttackCount": ("is_other_attack", "sum"),
        "RowCount": (feature_name, "count"),
    }
    if port_flag:
        port_aggs = {
            "is_port21": ("is_port21", "max"),
            "is_port22": ("is_port22", "max"),
            "is_port20": ("is_port20", "max")
        }
        agg_dict = {**agg_dict, **port_aggs}
        
    grouped = df.resample("1s").agg(**agg_dict)
    grouped = grouped[grouped["RowCount"] > 0]
    return grouped, "FeatureMean"
