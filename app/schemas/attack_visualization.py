from pydantic import BaseModel
from typing import List, Optional, Union

class DataSchema(BaseModel):
    """
    Defines the 'data' portion of the time-series response:
      - timestamps: parallel array of ISO8601 date/time strings
      - values: parallel array of numeric values
      - attackMarkPoint: list of [timestamp, value] pairs for user-specified attacks
      - otherAttackMarkPoint: list of [timestamp, value] pairs for 'other' attacks
      - feature: e.g. 'Flow Bytes/s' or another selected feature
    """
    timestamps: List[str]
    values: List[float]
    attackMarkPoint: List[List[Union[str, float]]]
    otherAttackMarkPoint: List[List[Union[str, float]]]
    feature: str

class HighlightItem(BaseModel):
    """
    Represents one item in a highlight pair.
    If it's the first item in the pair, 'name' can be your attack type
    or 'otherAttack'. If it's the second item, 'xAxis' remains.
    """
    name: Optional[str] = None
    xAxis: str

class PartitionBoundary(BaseModel):
    """
    Represents a partition boundary,
    specifying start and end timestamps for that partition.
    """
    start: str
    end: str

class GetTimeSeriesAttackDataResponse(BaseModel):
    """
    The overall response shape, now including partitions:

      {
        "data": { ...DataSchema fields... },
        "highlight": [
          [
            {"name": "<attackType>", "xAxis": "<startTimestamp>"},
            {"xAxis": "<endTimestamp>"}
          ],
          ...
        ],
        "partitions": [
          {
            "start": "<ISO date/time of partition start>",
            "end": "<ISO date/time of partition end>"
          },
          ...
        ]
      }
    """
    data: DataSchema
    highlight: List[List[HighlightItem]]
    partitions: List[PartitionBoundary] = []
