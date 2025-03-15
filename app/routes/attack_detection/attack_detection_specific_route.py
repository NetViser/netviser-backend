from collections import Counter
from fastapi import APIRouter, Depends, Cookie, Response
from typing import Optional
import numpy as np
import io

from app.services.input_handle_service import preprocess
from app.services.s3_service import S3
from app.services.redis_service import RedisClient
from app.configs.config import get_settings

router = APIRouter(
    prefix="/api/attack-detection/specific", tags=["specific attack detection"]
)
settings = get_settings()
redis_client = RedisClient()

sankey_fields = [
    "srcPort",
    "dstPort",
    "srcIp",
    "srcIpPortPairCount",
    "portPairCount",
]

# Define field mapping for specific attack types (camelCase keys)
attack_field_mapping = {
    "FTP-Patator": [
        "flowBytesPerSecond",
        "totalTCPFlowTime",
        "bwdIATMean",
        *sankey_fields,
    ],
    "SSH-Patator": [
        "totalTCPFlowTime",
        "bwdInitWinBytes",
        "fwdPacketLengthMax",
        *sankey_fields,
    ],
    "DDoS": [
        "dstIp",
        *sankey_fields,
        "packetlengthmean",
        "flowDuration",
        "flowPacketsPerSecond",
        "bwdpacketlengthstd",
        "protocol",
    ],
    "Portscan": [
        *sankey_fields,
        "dstIp",
        "totalLengthOfFwdPacket",
    ],
    "DoS Hulk": [
        "bwdpacketlengthstd",
        "bwdInitWinBytes",
        "fwdPacketLengthMax",
        "bwdIATMean",
        "dstIp",
        *sankey_fields,
    ],
    "DoS Slowloris": [
        "totalTCPFlowTime",
        "bwdIATMean",
        *sankey_fields,
        "fwdPSHFlags",
    ],
}

# Full list of fields if the attack type is not one of the above
all_fields = [
    "timestamp",
    "flowBytesPerSecond",
    "flowDuration",
    "flowPacketsPerSecond",
    "averagePacketSize",
    "totalFwdPacket",
    "totalBwdPacket",
    "totalLengthOfFwdPacket",
    "totalTCPFlowTime",
    "bwdpacketlengthstd",
    "bwdIATMean",
    "bwdInitWinBytes",
    "fwdPacketLengthMax",
    "fwdPSHFlags",
    "protocol",
    "srcIp",
    "dstIp",
    "srcPort",
    "dstPort",
    "portPairCount",
    "srcIpPortPairCount",
    "packetlengthmean",
    "synflagcount",
    "ackflagcount",
    "subflowfwdbytes",
    "protocol_distribution",
]


def build_record(row: dict, field_list: list, protocol_distribution: dict) -> dict:
    """
    Convert a DataFrame row (as a dict) to a new dict with camelCase keys.
    The mapping below defines how each output key is derived from the DataFrame.
    """
    mapping = {
        "timestamp": lambda r: r["Timestamp"].isoformat(),
        "flowBytesPerSecond": lambda r: r["Flow Bytes/s"],
        "flowDuration": lambda r: r["Flow Duration"],
        "flowPacketsPerSecond": lambda r: r["Flow Packets/s"],
        "averagePacketSize": lambda r: r["Average Packet Size"],
        "totalFwdPacket": lambda r: r["Total Fwd Packet"],
        "totalBwdPacket": lambda r: r["Total Bwd packets"],
        "totalLengthOfFwdPacket": lambda r: r["Total Length of Fwd Packet"],
        "totalTCPFlowTime": lambda r: r["Total TCP Flow Time"],
        "bwdpacketlengthstd": lambda r: r["Bwd Packet Length Std"],
        "bwdIATMean": lambda r: r["Bwd IAT Mean"],
        "bwdInitWinBytes": lambda r: r["Bwd Init Win Bytes"],
        "fwdPacketLengthMax": lambda r: r["Fwd Packet Length Max"],
        "fwdPSHFlags": lambda r: r["Fwd PSH Flags"],
        "protocol": lambda r: r["Protocol"],
        "srcIp": lambda r: r["Src IP"],
        "dstIp": lambda r: r["Dst IP"],
        "srcPort": lambda r: r["Src Port"],
        "dstPort": lambda r: r["Dst Port"],
        "portPairCount": lambda r: r["Port Pair Count"],
        "srcIpPortPairCount": lambda r: r["Src IP Port Pair Count"],
        "packetlengthmean": lambda r: r["Packet Length Mean"],
        "synflagcount": lambda r: r["SYN Flag Count"],
        "ackflagcount": lambda r: r["ACK Flag Count"],
        "subflowfwdbytes": lambda r: r["Subflow Fwd Bytes"],
        "protocol_distribution": lambda r: protocol_distribution,
    }
    return {key: mapping[key](row) for key in field_list if key in mapping}


@router.get("/")
async def get_specific_attack_detection(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
    """
    Retrieve the data for a specific attack type stored in S3 based on the session_id.
    Paginated results are returned for both normal and attack data.
    Only selected fields are returned if the attack type is recognized.
    """
    if not session_id:
        return Response(status_code=400, content="Session ID missing")

    try:
        session_data = redis_client.get_session_data(session_id)
        if not session_data:
            return Response(status_code=400, content="Session Expired or not found")

        file_data = await s3_service.read(session_data)
        file_like_object = io.BytesIO(file_data)

        data_frame = await preprocess(file_like_object)
        data_frame.reset_index(inplace=True)

        # Convert units from microseconds to seconds
        fields_to_convert = [
            "Flow Duration",      # Maps to flowDuration
            "Bwd IAT Mean",       # Maps to bwdIATMean
            "Total TCP Flow Time" # Maps to totalTCPFlowTime
        ]
        for field in fields_to_convert:
            if field in data_frame.columns:
                data_frame[field] = data_frame[field].astype(float) / 1_000_000

        # Round specified fields to 2 decimal places and apply log scale
        fields_to_round = [
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow Duration",
            "Average Packet Size",
        ]
        for field in fields_to_round:
            data_frame[field] = data_frame[field].astype(float).round(2)
            data_frame[field] = np.log10(data_frame[field] + 1)

        # Count the occurrences for each Src Port/Dst Port and Src IP/ Src Port pair
        port_pair_counts = (
            data_frame.groupby(["Src Port", "Dst Port"])
            .size()
            .reset_index(name="Port Pair Count")
        )
        src_ip_port_pair_counts = (
            data_frame.groupby(["Src IP", "Src Port"])
            .size()
            .reset_index(name="Src IP Port Pair Count")
        )

        # Merge the counts back into the DataFrame
        data_frame = data_frame.merge(
            port_pair_counts, on=["Src Port", "Dst Port"], how="left"
        )
        data_frame = data_frame.merge(
            src_ip_port_pair_counts, on=["Src IP", "Src Port"], how="left"
        )

        # Separate data into normal and attack DataFrames
        normal_df = data_frame[data_frame["Label"] == "BENIGN"].sort_values(
            by="Timestamp", ascending=False
        )
        attack_df = data_frame[data_frame["Label"] == attack_type].sort_values(
            by="Timestamp", ascending=False
        )

        protocol_distribution_normal = Counter(normal_df["Protocol"])
        protocol_distribution_attack = Counter(attack_df["Protocol"])

        # Determine which fields to return based on the attack_type.
        # If the attack type is not recognized, return all fields.
        if attack_type in attack_field_mapping:
            selected_fields = attack_field_mapping[attack_type]
        else:
            selected_fields = all_fields

        # Build records for normal and attack data using the selected fields
        normal_data = [
            build_record(row, selected_fields, dict(protocol_distribution_normal))
            for row in normal_df.dropna().to_dict(orient="records")
        ]
        attack_data = [
            build_record(row, selected_fields, dict(protocol_distribution_attack))
            for row in attack_df.dropna().to_dict(orient="records")
        ]

        return {
            "normalData": normal_data,
            "attackData": attack_data,
        }

    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve data.")
