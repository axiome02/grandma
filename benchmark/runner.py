"""
runner.py — Main benchmark entry point.

Usage:
    python benchmark/runner.py
    python benchmark/runner.py --limit 10
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.models import AnthropicClient, GeminiClient, OpenAIClient

PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GeminiClient,
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_dataset_hash(dataset_path: str) -> str:
    with open(dataset_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def resolve_image_path(question: dict, dataset_path: str) -> str | None:
    if not question.get("image"):
        return None
    base = Path(dataset_path).parent
    return str(base / question["image"])


def run_model(client, questions: list[dict], dataset_path: str, limit: int | None) -> list[dict]:
    results = []
    subset = questions[:limit] if limit else questions

    for i, q in enumerate(subset, 1):
        image_path = resolve_image_path(q, dataset_path)
        print(f"  [{i}/{len(subset)}] {q['id']} ({q['difficulty']}) — {q['type']}", end="", flush=True)
        try:
            response = client.query(q["question"], image_path=image_path)
            status = "ok"
        except Exception as e:
            response = f"ERROR: {e}"
            status = "error"

        print(f" ✓" if status == "ok" else f" ✗")
        results.append({
            "id": q["id"],
            "type": q["type"],
            "difficulty": q["difficulty"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer": q["answer"],
            "model_response": response,
            "status": status,
            "score": None,  # Filled in by evaluator.py
        })

    return results


def save_results(results: list[dict], model_name: str, config: dict, dataset_hash: str, output_dir: str):
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(output_dir) / date_str
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "-")
    filepath = run_dir / f"{safe_name}.json"

    output = {
        "meta": {
            "model": model_name,
            "dataset": config["dataset"],
            "dataset_hash": dataset_hash,
            "run_date": datetime.now().isoformat(),
            "total_questions": len(results),
        },
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  → Saved: {filepath}")
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Run the LLM Logic Benchmark")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    parser.add_argument("--models", nargs="+", default=None, help="Override models to run (by name)")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = config["dataset"]
    questions = load_dataset(dataset_path)
    dataset_hash = compute_dataset_hash(dataset_path)

    print(f"\n🧠 LLM Logic Benchmark")
    print(f"   Dataset : {dataset_path} (hash: {dataset_hash})")
    print(f"   Questions: {len(questions)}" + (f" (limited to {args.limit})" if args.limit else ""))
    print()

    enabled_models = [m for m in config["models"] if m.get("enabled", True)]
    if args.models:
        enabled_models = [m for m in enabled_models if m["name"] in args.models]

    if not enabled_models:
        print("❌ No enabled models found. Check your config.yaml.")
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
            print(f"  ✗ Failed to initialize client: {e}")
            continue

        results = run_model(client, questions, dataset_path, args.limit)
        save_results(results, name, config, dataset_hash, output_dir)
        print()

    print("✅ Benchmark complete. Run evaluator.py to score the results.")


if __name__ == "__main__":
    main()
