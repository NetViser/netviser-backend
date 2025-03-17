from collections import Counter
from fastapi import APIRouter, Depends, Cookie, Response
from typing import Dict, Optional
import numpy as np
import io

from app.schemas.overview_visualization import AttackDetectionResponse
from app.services.input_handle_service import preprocess
from app.services.s3_service import S3
from app.services.redis_service import RedisClient
from app.configs.config import get_settings

router = APIRouter(prefix="/api/attack-detection", tags=["specific attack detection"])
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


def generate_sankey_data(attack_data: list, slice_count: int = 25) -> dict:
    """Generate Sankey diagram data from attack records."""
    attack_data = attack_data[:slice_count]

    node_mapping = {}
    nodes_set = set()
    for record in attack_data:
        src_ip, src_port, dst_port = (
            str(record["srcIp"]),
            str(record["srcPort"]),
            str(record["dstPort"]),
        )
        nodes_set.add(src_ip)
        nodes_set.add(src_port)
        nodes_set.add(dst_port)
        if src_ip not in node_mapping:
            node_mapping[src_ip] = "Source IP"
        if src_port not in node_mapping:
            node_mapping[src_port] = "Source Port"
        if dst_port not in node_mapping:
            node_mapping[dst_port] = "Dst Port"

    nodes = [{"name": name} for name in nodes_set]
    src_ip_links = [
        {
            "source": str(r["srcIp"]),
            "target": str(r["srcPort"]),
            "value": r["srcIpPortPairCount"],
        }
        for r in attack_data
    ]
    port_links = [
        {
            "source": str(r["srcPort"]),
            "target": str(r["dstPort"]),
            "value": r["portPairCount"],
        }
        for r in attack_data
    ]

    return {
        "nodes": nodes,
        "links": src_ip_links + port_links,
        "nodeMapping": node_mapping,
    }


def generate_sankey_data(attack_data: list, slice_count: int = 25) -> dict:
    """Generate Sankey diagram data from attack records."""
    attack_data = attack_data[:slice_count]

    node_mapping = {}
    nodes_set = set()
    for record in attack_data:
        src_ip, src_port, dst_port = (
            str(record["srcIp"]),
            str(record["srcPort"]),
            str(record["dstPort"]),
        )
        nodes_set.add(src_ip)
        nodes_set.add(src_port)
        nodes_set.add(dst_port)
        if src_ip not in node_mapping:
            node_mapping[src_ip] = "Source IP"
        if src_port not in node_mapping:
            node_mapping[src_port] = "Source Port"
        if dst_port not in node_mapping:
            node_mapping[dst_port] = "Dst Port"

    nodes = [{"name": name} for name in nodes_set]
    src_ip_links = [
        {
            "source": str(r["srcIp"]),
            "target": str(r["srcPort"]),
            "value": r["srcIpPortPairCount"],
        }
        for r in attack_data
    ]
    port_links = [
        {
            "source": str(r["srcPort"]),
            "target": str(r["dstPort"]),
            "value": r["portPairCount"],
        }
        for r in attack_data
    ]

    return {
        "nodes": nodes,
        "links": src_ip_links + port_links,
        "nodeMapping": node_mapping,
    }


def compute_means(data: list, fields: list) -> Dict[str, float]:
    """Compute mean values for specified numeric fields."""
    means = {}
    for field in fields:
        if field in [
            "srcIp",
            "dstIp",
            "srcPort",
            "dstPort",
            "protocol_distribution",
            *sankey_fields,
        ]:
            continue  # Skip non-numeric or Sankey-specific fields
        values = [
            record[field]
            for record in data
            if field in record and isinstance(record[field], (int, float))
        ]
        means[field] = float(np.mean(values).round(2)) if values else 0.0
    return means


@router.get("/overview", response_model=AttackDetectionResponse)
async def get_specific_attack_detection(
    attack_type: str,
    session_id: Optional[str] = Cookie(None),
    s3_service: S3 = Depends(S3),
):
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
        fields_to_convert = ["Flow Duration", "Bwd IAT Mean", "Total TCP Flow Time"]
        for field in fields_to_convert:
            if field in data_frame.columns:
                data_frame[field] = data_frame[field].astype(float) / 1_000_000

        # Round and log-scale specified fields
        fields_to_round = [
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow Duration",
            "Average Packet Size",
        ]
        for field in fields_to_round:
            data_frame[field] = data_frame[field].astype(float).round(2)
            data_frame[field] = np.log10(data_frame[field] + 1)

        # Calculate pair counts
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
        data_frame = data_frame.merge(
            port_pair_counts, on=["Src Port", "Dst Port"], how="left"
        )
        data_frame = data_frame.merge(
            src_ip_port_pair_counts, on=["Src IP", "Src Port"], how="left"
        )

        # Separate normal and attack data
        normal_df = data_frame[data_frame["Label"] == "BENIGN"].sort_values(
            by="Timestamp", ascending=False
        )
        attack_df = data_frame[data_frame["Label"] == attack_type].sort_values(
            by="Timestamp", ascending=False
        )

        # Determine fields based on attack type
        selected_fields = attack_field_mapping.get(attack_type, all_fields)

        # Build records
        protocol_distribution_normal = Counter(normal_df["Protocol"])
        protocol_distribution_attack = Counter(attack_df["Protocol"])
        normal_data = [
            build_record(row, selected_fields, dict(protocol_distribution_normal))
            for row in normal_df.dropna().to_dict(orient="records")
        ]
        attack_data = [
            build_record(row, selected_fields, dict(protocol_distribution_attack))
            for row in attack_df.dropna().to_dict(orient="records")
        ]

        # Precompute visualization data
        normal_means = compute_means(normal_data, selected_fields)
        attack_means = compute_means(attack_data, selected_fields)
        sankey_data = generate_sankey_data(attack_data)

        # Compute unique source IPs
        normal_unique_src_ips = len(set(r["srcIp"] for r in normal_data))
        attack_unique_src_ips = len(set(r["srcIp"] for r in attack_data))

        # Compute unique source ports
        normal_unique_src_ports = len(set(r["srcPort"] for r in normal_data))
        attack_unique_src_ports = len(set(r["srcPort"] for r in attack_data))

        # Compute unique destination ports
        normal_unique_dst_ports = len(set(r["dstPort"] for r in normal_data))
        attack_unique_dst_ports = len(set(r["dstPort"] for r in attack_data))

        return {
            "means": {
                "normal": normal_means,
                "attack": attack_means,
            },
            "sankeyData": sankey_data,
            "uniqueSrcIps": {
                "normal": normal_unique_src_ips,
                "attack": attack_unique_src_ips,
            },
            "uniqueSrcPorts": {
                "normal": normal_unique_src_ports,
                "attack": attack_unique_src_ports,
            },
            "uniqueDstPorts": {
                "normal": normal_unique_dst_ports,
                "attack": attack_unique_dst_ports,
            },
        }

    except Exception as e:
        print(e)
        return Response(status_code=400, content="Failed to retrieve data.")
