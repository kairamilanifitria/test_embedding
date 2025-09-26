from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

# Use a smaller model first to fit Railway free plan
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed(req: EmbedRequest):
    # Tokenize batch of texts
    inputs = tokenizer(req.texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token representation
    embeddings = outputs.last_hidden_state[:, 0, :].numpy().tolist()

    return {"embeddings": embeddings}
