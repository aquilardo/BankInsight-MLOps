from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        allow_inf_nan=False
    )

    age: float = Field(ge=17, le=100)
    job: str = Field(min_length=1)
    marital: str = Field(min_length=1)
    education: str = Field(min_length=1)
    default: str = Field(min_length=1)
    housing: str = Field(min_length=1)
    loan: str = Field(min_length=1)
    contact: str = Field(min_length=1)
    month: str = Field(min_length=1)
    day_of_week: str = Field(min_length=1)

    campaign: float = Field(ge=1)
    pdays: float = Field(ge=0)
    previous: float = Field(ge=0)
    poutcome: str = Field(min_length=1)

    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")


class PredictionResponse(BaseModel):
    prediction: str
    model_name: str
    model_version: str
    model_alias: str
