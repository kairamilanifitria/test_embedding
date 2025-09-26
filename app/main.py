from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Railway Test API")

# A simple dummy "model"
def tiny_model(x: float) -> float:
    return x * 2  # just doubles the input

class InputData(BaseModel):
    number: float

@app.get("/")
def read_root():
    return {"message": "Hello from Railway Free Tier!"}

@app.post("/predict")
def predict(data: InputData):
    result = tiny_model(data.number)
    return {"input": data.number, "prediction": result}
