from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
import torch
import uuid
import time
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Change Model Path and Name here
MODEL_PATH = "../meta-llama/Llama-3-1-8B-Instruct"
MODEL_NAME = "llama-3.1-8b-instruct"

app = FastAPI()

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stop: Optional[List[str]] = None

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("Loading tokenizer...")
    model_name = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    print("Model loaded.")

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    prompt = build_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(tokenizer(answer)["input_ids"]),
            "total_tokens": len(inputs.input_ids[0]) + len(tokenizer(answer)["input_ids"])
        }
    }

@app.post("/v1/completions")
async def completions(req: dict):
    if "prompt" not in req:
        raise HTTPException(status_code=422, detail="Missing 'prompt' field")

    prompt = req["prompt"]
    temperature = max(req.get("temperature", 0), 1e-5)
    max_tokens = req.get("max_tokens", 512)
    model_name = req.get("model", MODEL_NAME)
    stop_strs = req.get("stop", ["User:", "\nUser:", "\n\nUser:"])

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Slice off the prompt
    answer = full_response[len(prompt):].strip()

    # Truncate on stop string
    for stop in stop_strs:
        if stop in answer:
            answer = answer.split(stop)[0].strip()
            break

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "text": answer,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(tokenizer(answer)["input_ids"]),
            "total_tokens": len(inputs.input_ids[0]) + len(tokenizer(answer)["input_ids"])
        }
    }

def build_prompt(messages: List[Message]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"[INST] <<SYS>>\n{msg.content}\n<</SYS>>\n"
        elif msg.role == "user":
            prompt += f"[INST] {msg.content} [/INST]\n"
        elif msg.role == "assistant":
            prompt += f"{msg.content}\n"
    return prompt

if __name__ == "__main__":
    uvicorn.run("llama_server:app", host="0.0.0.0", port=8010)
