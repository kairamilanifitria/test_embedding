from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Tiny GPT-2 API")

# Lightweight CPU model
model_name = "sshleifer/tiny-gpt2"
generator = pipeline("text-generation", model=model_name, device=-1)  # CPU only

class RequestBody(BaseModel):
    prompt: str
    max_length: int = 50

@app.get("/")
def read_root():
    return {"message": "Tiny GPT-2 API is running!"}

@app.post("/generate")
def generate_text(request: RequestBody):
    result = generator(request.prompt, max_length=request.max_length, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}
