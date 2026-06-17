from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from house_price.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)


class HouseData(BaseModel):
    LotFrontage: float
    LotArea: float
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    MasVnrArea: float
    BsmtFinSF1: float
    BsmtUnfSF: float
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    GrLivArea: float
    FullBath: int
    GarageCars: int
    GarageArea: float

    MSZoning: str
    Street: str
    LotShape: str
    Neighborhood: str


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.post("/predict")
def predict(data: HouseData):

    input_df = pd.DataFrame([{
        "LotFrontage": data.LotFrontage,
        "LotArea": data.LotArea,
        "OverallQual": data.OverallQual,
        "OverallCond": data.OverallCond,
        "YearBuilt": data.YearBuilt,
        "YearRemodAdd": data.YearRemodAdd,
        "MasVnrArea": data.MasVnrArea,
        "BsmtFinSF1": data.BsmtFinSF1,
        "BsmtUnfSF": data.BsmtUnfSF,
        "TotalBsmtSF": data.TotalBsmtSF,
        "1stFlrSF": data.FirstFlrSF,
        "2ndFlrSF": data.SecondFlrSF,
        "GrLivArea": data.GrLivArea,
        "FullBath": data.FullBath,
        "GarageCars": data.GarageCars,
        "GarageArea": data.GarageArea,
        "MSZoning": data.MSZoning,
        "Street": data.Street,
        "LotShape": data.LotShape,
        "Neighborhood": data.Neighborhood
    }])

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(input_df)

    return {
        "predicted_price": round(float(prediction), 2)
    }