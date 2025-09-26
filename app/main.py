from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

app = FastAPI(title="Tiny GPT-2 ONNX API")

# Load tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load ONNX model
onnx_model_path = "app/tiny-gpt2.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

class RequestBody(BaseModel):
    prompt: str
    max_length: int = 50

@app.get("/")
def read_root():
    return {"message": "Tiny GPT-2 ONNX API is running!"}

@app.post("/generate")
def generate_text(request: RequestBody):
    # Tokenize input
    inputs = tokenizer(request.prompt, return_tensors="np")
    input_ids = inputs["input_ids"]

    # Prepare output
    output_ids = input_ids.copy()

    # Simple greedy generation loop
    for _ in range(request.max_length):
        ort_inputs = {"input_ids": output_ids}
        logits = session.run(None, ort_inputs)[0]

        # Pick the last token's logits
        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])

        # Append predicted token
        output_ids = np.concatenate([output_ids, [[next_token_id]]], axis=1)

        # Stop if EOS token
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
