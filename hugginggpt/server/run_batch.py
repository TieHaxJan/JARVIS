import json
import os
import argparse
import time
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import requests
import yaml

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return cfg


def load_prompts_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of {"prompt": "..."} items
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def slice_prompts(prompts: list[dict], start_at_1based: int, limit: int | None) -> tuple[list[dict], int]:
    start_index = max(start_at_1based - 1, 0)
    sliced = prompts[start_index:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced, start_index


def ensure_empty_file_once(path: Path) -> None:
    if path.exists():
        # Empty only ONCE at program start
        path.write_text("", encoding="utf-8")


def jsonl_append(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument("--config", type=str, default="configs/config.localllama.yaml",
                        help="YAML config containing hf_token, model_name")

    # Prompt files
    parser.add_argument("--prompts", type=str, default="test_prompts.json",
                        help="Base prompts JSON (used for iterations 1..N-1)")
    parser.add_argument("--rephrased-prompts", type=str, default="test_prompts_rephrased.json",
                        help="Rephrased prompts JSON (used for iteration N)")

    # Batch slicing
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of prompts to run (default: all)")
    parser.add_argument("--start-at", type=int, default=1,
                        help="1-based index to start processing (default: 1)")

    # Iterations
    parser.add_argument("--iterations", type=int, default=6,
                        help="How many iterations to run (default: 6)")

    # Output root
    parser.add_argument("--out-root", type=str, default="runs",
                        help="Root folder for run outputs")

    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    hf_token = cfg['huggingface']['token']
    model_name = cfg['model']
    server_url = "http://127.0.0.1:8004/hugginggpt"

    if not hf_token:
        raise SystemExit("HF token missing in config. Expected key: hf_token")
    if not model_name:
        raise SystemExit("Model name missing in config. Expected key: model_name")

    # Create run folder: runs/YYYYMMDD_HHMMSS_ab12/
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:4]
    run_dir = Path(args.out_root) / f"{run_stamp}_{short_id}"
    (run_dir / "public").mkdir(parents=True, exist_ok=True)
    (run_dir / "public" / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "public" / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "public" / "audios").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "explanations").mkdir(parents=True, exist_ok=True)

    # Clear explanations.jsonl ONCE at start if exists
    explanations_src = Path("data") / "explanations.jsonl"
    ensure_empty_file_once(explanations_src)

    # Prepare global batch output jsonl
    batch_output_path = run_dir / "batch_output.jsonl"
    if batch_output_path.exists():
        batch_output_path.unlink()

    total_start = time.time()
    total_errors = 0

    # Load both prompt sets once
    base_prompts_all = load_prompts_json(args.prompts)
    rephrased_prompts_all = load_prompts_json(args.rephrased_prompts)

    # Apply slicing consistently to both sets (so prompt_id aligns)
    base_prompts, start_index = slice_prompts(base_prompts_all, args.start_at, args.limit)
    rephrased_prompts, _ = slice_prompts(rephrased_prompts_all, args.start_at, args.limit)

    # Sanity: both should have same length for the slice
    if len(rephrased_prompts) != len(base_prompts):
        raise SystemExit(
            f"Sliced prompt counts differ: base={len(base_prompts)} rephrased={len(rephrased_prompts)}. "
            f"Make sure both files contain the same prompts in the same order."
        )

    total_items = len(base_prompts)
    if total_items == 0:
        print("No prompts to run after slicing.")
        return

    # Run iterations
    for iteration in range(1, args.iterations + 1):
        print(f"\n==============================")
        print(f"ITERATION {iteration}/{args.iterations}")
        print(f"Run folder: {run_dir}")
        print(f"==============================")

        # Choose prompts: last iteration uses rephrased prompts
        prompts = rephrased_prompts if iteration == args.iterations else base_prompts

        iter_errors = 0
        iter_start = time.time()

        for local_idx, item in enumerate(prompts, start=1):
            # Keep original "global" prompt index based on start_at offset
            prompt_id = start_index + local_idx  # 1-based

            item_start = time.time()
            prompt = item.get("prompt", "")

            print(f"\n=== Prompt {prompt_id}/{start_index + total_items} | iter {iteration} ===")
            print(f"Prompt: {prompt}")

            wrapped_prompt = (
                f"<<PROMPT_ID:{prompt_id}>>"
                f"<<ITERATION:{iteration}>>"
                f"<<RUN_DIR:{str(run_dir)}>>\n"
                f"{prompt}"
            )

            payload = {
                "messages": [{"role": "user", "content": wrapped_prompt}],
                "api_type": "huggingface",
                "api_key": hf_token,
                "api_endpoint": "",
                "model": model_name,
            }

            try:
                resp = requests.post(server_url, json=payload, timeout=600)
                resp.raise_for_status()
                response = resp.json()
            except Exception as e:
                duration = time.time() - item_start
                iter_errors += 1
                total_errors += 1

                record = {
                    "prompt_id": prompt_id,
                    "iteration": iteration,
                    "prompt": prompt,
                    "response": f"ERROR: {e}",
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                }
                jsonl_append(batch_output_path, record)

                print(f"ERROR for prompt {prompt_id}: {e}")
                continue

            duration = time.time() - item_start
            message = response.get("message", "")

            record = {
                "prompt_id": prompt_id,
                "iteration": iteration,
                "prompt": prompt,
                "response": message,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
            }

            jsonl_append(batch_output_path, record)

            print(f"Response received in {duration:.2f}s")
            print(f"Response: {message}")

        # After each iteration, copy explanations.jsonl into run_dir/explanations/
        # (Even if it's empty, we copy so the run folder is self-contained.)
        copied_path = run_dir / "explanations" / f"iteration_{iteration}_explanations.jsonl"
        if explanations_src.exists():
            shutil.copyfile(explanations_src, copied_path)
        else:
            # Create empty placeholder if source doesn't exist
            copied_path.write_text("", encoding="utf-8")

        iter_duration = time.time() - iter_start
        print(f"\nIteration {iteration} runtime: {iter_duration:.2f}s")
        print(f"Iteration {iteration} request errors: {iter_errors}")

    total_duration = time.time() - total_start
    print("\n=== DONE ===")
    print(f"Run folder: {run_dir}")
    print(f"Total runtime: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Total request errors: {total_errors}")


if __name__ == "__main__":
    main()
