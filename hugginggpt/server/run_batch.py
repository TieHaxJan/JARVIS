import json
import requests
import os
import argparse
from datetime import datetime

SERVER_URL = "http://127.0.0.1:8004/hugginggpt"       # adjust if needed
HF_TOKEN = ""          

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"       

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Number of prompts to run (default: all)")
args = parser.parse_args()

# Load prompts
with open("test_prompts.json", "r") as f:
    prompts = json.load(f)

# Limit prompts
if args.limit is not None:
    prompts = prompts[:args.limit]

os.makedirs("batch_results", exist_ok=True)

results = []

# Run prompts
for i, item in enumerate(prompts, start=1):
    prompt = item["prompt"]

    print(f"\n=== {i}/{len(prompts)}  RUNNING ===")
    print(f"Prompt: {prompt}")

    payload = {
        "messages": [ {"role": "user", "content": prompt} ],
        "api_type": "huggingface",
        "api_key": HF_TOKEN,
        "api_endpoint": "",
        "model": MODEL_NAME
    }

    response = requests.post(SERVER_URL, json=payload).json()
    
    message = response.get("message", "")

    item["response"] = message
    print(f"Response: {message}")
    results.append(item)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"batch_results/batch_output_{timestamp}.json"

with open(output_path, "w") as out:
    json.dump(results, out, indent=2)

print("\n=== DONE, all items processed ===")
