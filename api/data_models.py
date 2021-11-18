from pydantic import BaseModel


class Request(BaseModel):
    day: int
    period: float
    nwsprice: float
    nwsdemand: float
    vicprice: float
    vicdemand: float
    transfer: float


class Response(BaseModel):
    timestamp: str
    class_pred: str
