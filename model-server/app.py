from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()

MODEL_NAME = "Qwen/Qwen3.5-2B"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"]
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.environ["HF_TOKEN"]
)


def get_device():
    if hasattr(model, "device"):
        return model.device
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Request(BaseModel):
    prompt: str


@app.post("/generate")
def generate(req: Request):
    device = get_device()

    inputs = tokenizer(
        req.prompt,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.35,
            top_p=0.85,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"text": text}