from typing import List, Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from runner import Runner

app = FastAPI()
runner = {}

class Train_Request(BaseModel):
    sequences: List[str]
    labels: List[str]
    epochs: int


class Predict_Request(BaseModel):
    sequences: Any

@app.on_event("startup")
async def startup_event():
    print("========== d up ===========")
    max_sequence = 128
    classes = 4
    model = "../models/bert_classify.h5"
    runner["runner"] = Runner(max_sequence, classes)
    print("========== End Start up ===========")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/train")
def train(request: Train_Request):
    runner["runner"].train(request.sequences, request.labels, request.epochs)
    return {"train": "Finished"}

@app.post("/prediction")
def predict(request: Predict_Request):
    predictions = runner["runner"].prediction(request.sequences)
    return {"predictions": str(predictions)}