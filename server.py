import pickle
from typing import Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse

with open("pipeline_v1.bin", "rb") as file_in:
    pipeline = pickle.load(file_in)

app = FastAPI()


def predict_single(customer: dict[str, Any]):
    result = pipeline.predict_proba([customer])
    return result[0, 1]


@app.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/predict")
def predict(client: dict[str, Any]):
    prob = predict_single(client)
    return {"prob": prob}
