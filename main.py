from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

# Load model once at startup
MODEL_NAME = "alibaba-nlp/gte-multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
def embed(req: EmbedRequest):
    # Tokenize input
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token representation (you can change pooling if needed)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy().tolist()

    return {"embedding": embeddings}
