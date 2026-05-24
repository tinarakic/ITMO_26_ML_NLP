from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    token=os.environ["HF_TOKEN"]
)

class Request(BaseModel):
    prompt: str


@app.post("/generate")
def generate(req: Request):

    inputs = tokenizer(
        req.prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"text": text}