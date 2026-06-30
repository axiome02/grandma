"""
runner.py — Main benchmark entry point.

Usage:
    python benchmark/runner.py
    python benchmark/runner.py --limit 5
    python benchmark/runner.py --models gpt-4o claude-3-5-sonnet-20241022
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.models import AnthropicClient, GeminiClient, OpenAIClient

PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GeminiClient,
}

SYSTEM_PROMPT = """You are taking a visual logic benchmark test.

You will be shown an image containing a math equation, and a question about it.

You MUST respond using this exact format — no exceptions:

Answer: [your final answer, as short and precise as possible]
Reasoning: [your step-by-step reasoning explaining how you arrived at the answer]

Do not add anything before or after this format."""


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_dataset_hash(dataset_path: str) -> str:
    with open(dataset_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def resolve_image_path(question: dict, dataset_path: str) -> str:
    base = Path(dataset_path).parent
    return str(base / question["image"])


def parse_response(raw: str) -> dict:
    """Extract Answer and Reasoning from the model's structured response."""
    answer = ""
    reasoning = ""
    for line in raw.splitlines():
        if line.lower().startswith("answer:"):
            answer = line[len("answer:"):].strip()
        elif line.lower().startswith("reasoning:"):
            reasoning = line[len("reasoning:"):].strip()
    return {"answer": answer, "reasoning": reasoning}


def run_model(client, questions: list[dict], dataset_path: str, limit: int | None) -> list[dict]:
    results = []
    subset = questions[:limit] if limit else questions

    for i, q in enumerate(subset, 1):
        image_path = resolve_image_path(q, dataset_path)
        print(f"  [{i}/{len(subset)}] {q['id']} ({q['difficulty']})", end="", flush=True)

        if not Path(image_path).exists():
            print(f" ⚠ Image not found: {image_path}")
            results.append(_empty_result(q, status="missing_image"))
            continue

        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {q['question']}"
            response = client.query(full_prompt, image_path=image_path)
            parsed = parse_response(response["text"])
            input_tokens  = response["input_tokens"]
            output_tokens = response["output_tokens"]
            total_tokens  = input_tokens + output_tokens
            status = "ok"
            print(f" ✓  ({total_tokens} tokens: {input_tokens}↑ {output_tokens}↓)")
        except Exception as e:
            parsed = {"answer": "", "reasoning": ""}
            input_tokens = output_tokens = total_tokens = 0
            status = "error"
            print(f" ✗ ({e})")

        results.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "category": q["category"],
            "question": q["question"],
            "image": q["image"],
            "expected_answer": q["answer"],
            "model_answer": parsed["answer"],
            "model_reasoning": parsed["reasoning"],
            "status": status,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
            },
            "answer_score": None,    # Filled by evaluator.py
            "reasoning_score": None, # Filled by evaluator.py
        })

    return results


def _empty_result(q: dict, status: str) -> dict:
    return {
        "id": q["id"],
        "difficulty": q["difficulty"],
        "category": q["category"],
        "question": q["question"],
        "image": q["image"],
        "expected_answer": q["answer"],
        "model_answer": "",
        "model_reasoning": "",
        "status": status,
        "tokens": {"input": 0, "output": 0, "total": 0},
        "answer_score": None,
        "reasoning_score": None,
    }


def save_results(results: list[dict], model_name: str, config: dict, dataset_hash: str, output_dir: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(output_dir) / date_str
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "-")
    filepath = run_dir / f"{safe_name}.json"

    total_input  = sum(r["tokens"]["input"]  for r in results)
    total_output = sum(r["tokens"]["output"] for r in results)

    output = {
        "meta": {
            "model": model_name,
            "dataset": config["dataset"],
            "dataset_hash": dataset_hash,
            "run_date": datetime.now().isoformat(),
            "total_questions": len(results),
            "total_tokens": {
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
        },
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  → Saved: {filepath}")
    print(f"  → Total tokens: {total_input + total_output} ({total_input} input, {total_output} output)")
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Run the LLM Logic Benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    parser.add_argument("--models", nargs="+", default=None, help="Override which models to run")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = config["dataset"]
    questions = load_dataset(dataset_path)
    dataset_hash = compute_dataset_hash(dataset_path)

    print(f"\n🧠 LLM Logic Benchmark — Visual Digit Swap Edition")
    print(f"   Dataset  : {dataset_path} (hash: {dataset_hash})")
    print(f"   Questions: {len(questions)}" + (f" (limited to {args.limit})" if args.limit else ""))
    print()

    enabled_models = [m for m in config["models"] if m.get("enabled", True)]
    if args.models:
        enabled_models = [m for m in enabled_models if m["name"] in args.models]

    if not enabled_models:
        print("❌ No enabled models found. Check config.yaml.")
        sys.exit(1)

    output_dir = config.get("output_dir", "results/")

    for model_cfg in enabled_models:
        name = model_cfg["name"]
        provider = model_cfg["provider"]
        print(f"▶ Running: {name}")

        if provider not in PROVIDERS:
            print(f"  ⚠ Unknown provider '{provider}', skipping.")
            continue

        try:
            client = PROVIDERS[provider](model_name=name)
        except Exception as e:
            print(f"  ✗ Failed to initialize: {e}")
            continue

        results = run_model(client, questions, dataset_path, args.limit)
        save_results(results, name, config, dataset_hash, output_dir)
        print()

    print("✅ Done. Run evaluator.py to score the results.")


if __name__ == "__main__":
    main()
