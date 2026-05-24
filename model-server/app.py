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

    if not req.prompt.strip():
        return {"text": ""}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional literary translator. "
            )
        },
        {
            "role": "user",
            "content": req.prompt
        }
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_input, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,              # deterministic output
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # IMPORTANT: remove prompt tokens from output
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"text": text.strip()}