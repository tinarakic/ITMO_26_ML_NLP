from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map={"": 0} if device == "cuda" else None,
    token=os.environ["HF_TOKEN"]
)

model.eval()

class Request(BaseModel):
    prompt: str


@app.post("/generate")
def generate(req: Request):

    inputs = tokenizer(
    req.prompt,
    return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            # temperature=0.35,
            top_p=0.85,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"text": text.strip()}
