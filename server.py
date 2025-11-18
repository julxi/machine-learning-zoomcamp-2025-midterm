import pickle
from typing import Any
from fastapi import FastAPI


with open("pipeline_v1.bin", "rb") as file_in:
    pipeline = pickle.load(file_in)

app = FastAPI()


def predict_single(customer):
    result = pipeline.predict_proba(customer)
    return result[0, 1]


@app.post("/predict")
def predict(client: dict[str, Any]):
    prob = predict_single(client)

    return {"prob": prob}
