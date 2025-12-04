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
parser.add_argument("--start-at", type=int, default=1,
                    help="1-based index to start processing (default: 1)")
args = parser.parse_args()

# Load prompts
with open("test_prompts.json", "r") as f:
    prompts = json.load(f)
    
# Apply --start-at (convert to 0-based index internally)
start_index = max(args.start_at - 1, 0)
prompts = prompts[start_index:]

# Limit prompts
if args.limit is not None:
    prompts = prompts[:args.limit]

os.makedirs("batch_results", exist_ok=True)

results = []

error_count = 0

# Run prompts
for i, item in enumerate(prompts, start=start_index + 1):
    prompt = item["prompt"]

    print(f"\n=== {i}/{start_index + len(prompts)}  RUNNING ===")
    print(f"Prompt: {prompt}")

    payload = {
        "messages": [ {"role": "user", "content": prompt} ],
        "api_type": "huggingface",
        "api_key": HF_TOKEN,
        "api_endpoint": "",
        "model": MODEL_NAME
    }

    try:
        response = requests.post(SERVER_URL, json=payload).json()
    except Exception as e:
        print(f"ERROR requesting server for item {i}: {e}")
        error_count += 1
        item["response"] = f"ERROR: {e}"
        results.append(item)
        continue

    message = response.get("message", "")
    item["response"] = message
    print(f"Response: {message}")
    results.append(item)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"batch_results/batch_output_{timestamp}.json"

with open(output_path, "w") as out:
    json.dump(results, out, indent=2)

print("\n=== DONE, all items processed ===")
print(f"Total request errors: {error_count}")
