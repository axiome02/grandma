"""
evaluator.py — Score benchmark results.

Usage:
    # Human evaluation (interactive CLI)
    python benchmark/evaluator.py --eval human --file results/2026-06-30/gpt-4o.json

    # LLM-as-a-judge (automatic)
    python benchmark/evaluator.py --eval llm --judge gpt-4o --file results/2026-06-30/gpt-4o.json

    # Evaluate all files in a directory
    python benchmark/evaluator.py --eval human --dir results/2026-06-30/
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

JUDGE_PROMPT = """You are a strict benchmark evaluator.

You will be given:
- A logic question
- The expected correct answer (ground truth)
- A model's response

Your task: decide if the model's response is correct.

Rules:
- "correct": The model's response captures the key idea of the expected answer, even if worded differently.
- "partial": The model shows partial understanding but misses key elements.
- "incorrect": The model's response is wrong or irrelevant.

Respond with ONLY one word: correct, partial, or incorrect.

---
Question: {question}
Expected answer: {expected_answer}
Model response: {model_response}
---
Your verdict:"""


def score_with_human(entry: dict) -> str:
    print("\n" + "=" * 60)
    print(f"[{entry['id']}] Difficulty: {entry['difficulty']} | Type: {entry['type']}")
    print(f"\nQuestion:\n  {entry['question']}")
    print(f"\nExpected answer:\n  {entry['expected_answer']}")
    print(f"\nModel response:\n  {entry['model_response']}")
    print()

    while True:
        verdict = input("Score [y=correct / n=incorrect / p=partial / s=skip]: ").strip().lower()
        if verdict == "y":
            return "correct"
        elif verdict == "n":
            return "incorrect"
        elif verdict == "p":
            return "partial"
        elif verdict == "s":
            return None
        else:
            print("  Please enter y, n, p, or s.")


def score_with_llm(entry: dict, judge_model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = JUDGE_PROMPT.format(
        question=entry["question"],
        expected_answer=entry["expected_answer"],
        model_response=entry["model_response"],
    )

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().lower()

    if verdict not in ("correct", "partial", "incorrect"):
        print(f"  ⚠ Unexpected judge response: '{verdict}', defaulting to 'incorrect'")
        return "incorrect"

    return verdict


def evaluate_file(filepath: str, mode: str, judge_model: str | None) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data["meta"]["model"]
    results = data["results"]
    already_scored = [r for r in results if r.get("score") is not None]

    print(f"\n📋 Evaluating: {model_name}")
    print(f"   File: {filepath}")
    print(f"   Mode: {mode}" + (f" (judge: {judge_model})" if judge_model else ""))
    print(f"   Progress: {len(already_scored)}/{len(results)} already scored")

    for entry in results:
        if entry.get("score") is not None:
            continue  # Skip already scored entries
        if entry.get("status") == "error":
            entry["score"] = "incorrect"
            continue

        if mode == "human":
            verdict = score_with_human(entry)
            if verdict is None:
                continue
        else:
            verdict = score_with_llm(entry, judge_model)
            print(f"  [{entry['id']}] → {verdict}")

        entry["score"] = verdict

    # Save back
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print summary
    scored = [r for r in results if r.get("score") is not None]
    correct = sum(1 for r in scored if r["score"] == "correct")
    partial = sum(1 for r in scored if r["score"] == "partial")
    incorrect = sum(1 for r in scored if r["score"] == "incorrect")

    print(f"\n✅ {model_name} — {correct} correct | {partial} partial | {incorrect} incorrect / {len(scored)} scored")
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("--eval", choices=["human", "llm"], required=True, help="Evaluation mode")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model for LLM mode")
    parser.add_argument("--file", default=None, help="Single result file to evaluate")
    parser.add_argument("--dir", default=None, help="Directory of result files to evaluate")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Provide --file or --dir")

    files = []
    if args.file:
        files.append(args.file)
    if args.dir:
        files.extend(str(p) for p in Path(args.dir).glob("*.json"))

    for filepath in sorted(files):
        evaluate_file(
            filepath,
            mode=args.eval,
            judge_model=args.judge if args.eval == "llm" else None,
        )

    print("\n🏁 Evaluation complete. Run leaderboard.py to update the README.")


if __name__ == "__main__":
    main()
