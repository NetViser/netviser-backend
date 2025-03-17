from pydantic import BaseModel, Field
from typing import Dict, List, Literal

# Schema for a Sankey node
class SankeyNode(BaseModel):
    name: str = Field(..., description="Name of the node (e.g., IP address or port)")

# Schema for a Sankey link
class SankeyLink(BaseModel):
    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    value: int = Field(..., description="Value representing the flow magnitude")

# Schema for Sankey data
class SankeyData(BaseModel):
    nodes: List[SankeyNode] = Field(..., description="List of nodes in the Sankey diagram")
    links: List[SankeyLink] = Field(..., description="List of links between nodes in the Sankey diagram")
    nodeMapping: Dict[str, Literal["Source IP", "Source Port", "Dst Port"]] = Field(
        ..., description="Mapping of node names to their types (Source IP, Source Port, Dst Port)"
    )

# Schema for the means section (normal or attack)
class MeansData(BaseModel):
    flowBytesPerSecond: float = Field(0.0, description="Mean flow bytes per second (log-scaled)")
    flowDuration: float = Field(0.0, description="Mean flow duration in seconds")
    flowPacketsPerSecond: float = Field(0.0, description="Mean flow packets per second (log-scaled)")
    averagePacketSize: float = Field(0.0, description="Mean average packet size (log-scaled)")
    totalFwdPacket: float = Field(0.0, description="Mean total forward packets")
    totalBwdPacket: float = Field(0.0, description="Mean total backward packets")
    totalLengthOfFwdPacket: float = Field(0.0, description="Mean total length of forward packets")
    totalTCPFlowTime: float = Field(0.0, description="Mean total TCP flow time in seconds")
    bwdpacketlengthstd: float = Field(0.0, description="Mean standard deviation of backward packet length")
    bwdIATMean: float = Field(0.0, description="Mean backward inter-arrival time in seconds")
    bwdInitWinBytes: float = Field(0.0, description="Mean backward initial window bytes")
    fwdPacketLengthMax: float = Field(0.0, description="Mean maximum forward packet length")
    fwdPSHFlags: float = Field(0.0, description="Mean forward PSH flags")
    protocol: float = Field(0.0, description="Mean protocol number")
    packetlengthmean: float = Field(0.0, description="Mean packet length")
    synflagcount: float = Field(0.0, description="Mean SYN flag count")
    ackflagcount: float = Field(0.0, description="Mean ACK flag count")
    subflowfwdbytes: float = Field(0.0, description="Mean subflow forward bytes")

    class Config:
        extra = "allow"

# Schema for the full response
class AttackDetectionResponse(BaseModel):
    means: Dict[Literal["normal", "attack"], MeansData] = Field(
        ..., description="Mean values for numeric fields, separated by normal and attack data"
    )
    sankeyData: SankeyData = Field(..., description="Precomputed Sankey diagram data for attack records")
    uniqueSrcIps: Dict[Literal["normal", "attack"], int] = Field(
        ..., description="Count of unique source IPs for normal and attack data"
    )
    uniqueDstPorts: Dict[Literal["normal", "attack"], int] = Field(
        ..., description="Count of unique destination ports for normal and attack data"
    )
    uniqueSrcPorts: Dict[Literal["normal", "attack"], int] = Field(
        ..., description="Count of unique source ports for normal and attack data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "means": {
                    "normal": {
                        "flowBytesPerSecond": 5.23,
                        "totalTCPFlowTime": 0.12,
                        "bwdIATMean": 0.03,
                    },
                    "attack": {
                        "flowBytesPerSecond": 7.89,
                        "totalTCPFlowTime": 0.45,
                        "bwdIATMean": 0.09,
                    },
                },
                "sankeyData": {
                    "nodes": [{"name": "192.168.1.1"}, {"name": "22"}, {"name": "80"}],
                    "links": [
                        {"source": "192.168.1.1", "target": "22", "value": 10},
                        {"source": "22", "target": "80", "value": 15},
                    ],
                    "nodeMapping": {
                        "192.168.1.1": "Source IP",
                        "22": "Source Port",
                        "80": "Dst Port",
                    },
                },
                "uniqueSrcIps": {
                    "normal": 5,
                    "attack": 20,
                },
            }
        }