"""
evaluator.py — Score benchmark results (answer + reasoning).

Usage:
    python benchmark/evaluator.py --eval human --file results/2026-06-30/gpt-4o.json
    python benchmark/evaluator.py --eval llm --judge gpt-4o --file results/2026-06-30/gpt-4o.json
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

JUDGE_PROMPT = """You are a strict benchmark evaluator for a visual logic test.

You will receive:
- The question (based on an image)
- The expected correct answer (ground truth)
- The model's answer
- The model's reasoning

Evaluate TWO things separately:

1. ANSWER: Is the model's answer correct?
   - "correct": matches the expected answer (exact or equivalent meaning)
   - "incorrect": wrong or irrelevant

2. REASONING: Is the model's reasoning sound and coherent?
   - "valid": the reasoning logically leads to the answer and shows genuine understanding
   - "flawed": the reasoning is wrong, circular, or doesn't match the answer
   - "missing": no reasoning was provided

Respond with ONLY this exact format:
answer_score: correct|incorrect
reasoning_score: valid|flawed|missing

---
Question: {question}
Expected answer: {expected_answer}
Model answer: {model_answer}
Model reasoning: {model_reasoning}
---"""


def score_with_human(entry: dict) -> tuple[str, str] | None:
    print("\n" + "=" * 60)
    print(f"[{entry['id']}] Difficulty: {entry['difficulty']} | Category: {entry['category']}")
    print(f"\nQuestion:\n  {entry['question']}")
    print(f"\nExpected answer:\n  {entry['expected_answer']}")
    print(f"\nModel answer:\n  {entry['model_answer'] or '(empty)'}")
    print(f"\nModel reasoning:\n  {entry['model_reasoning'] or '(empty)'}")
    print()

    # Score the answer
    while True:
        a = input("Answer correct? [y=correct / n=incorrect / s=skip]: ").strip().lower()
        if a == "y":
            answer_score = "correct"
            break
        elif a == "n":
            answer_score = "incorrect"
            break
        elif a == "s":
            return None
        else:
            print("  Please enter y, n, or s.")

    # Score the reasoning
    while True:
        r = input("Reasoning valid? [v=valid / f=flawed / m=missing]: ").strip().lower()
        if r == "v":
            reasoning_score = "valid"
            break
        elif r == "f":
            reasoning_score = "flawed"
            break
        elif r == "m":
            reasoning_score = "missing"
            break
        else:
            print("  Please enter v, f, or m.")

    return answer_score, reasoning_score


def score_with_llm(entry: dict, judge_model: str) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = JUDGE_PROMPT.format(
        question=entry["question"],
        expected_answer=entry["expected_answer"],
        model_answer=entry["model_answer"] or "(no answer)",
        model_reasoning=entry["model_reasoning"] or "(no reasoning)",
    )

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )

    raw = response.choices[0].message.content.strip().lower()
    answer_score = "incorrect"
    reasoning_score = "missing"

    for line in raw.splitlines():
        if line.startswith("answer_score:"):
            val = line.split(":", 1)[1].strip()
            if val in ("correct", "incorrect"):
                answer_score = val
        elif line.startswith("reasoning_score:"):
            val = line.split(":", 1)[1].strip()
            if val in ("valid", "flawed", "missing"):
                reasoning_score = val

    return answer_score, reasoning_score


def evaluate_file(filepath: str, mode: str, judge_model: str | None) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data["meta"]["model"]
    results = data["results"]
    already_scored = [r for r in results if r.get("answer_score") is not None]

    print(f"\n📋 Evaluating: {model_name}")
    print(f"   File   : {filepath}")
    print(f"   Mode   : {mode}" + (f" (judge: {judge_model})" if judge_model else ""))
    print(f"   Progress: {len(already_scored)}/{len(results)} already scored")

    for entry in results:
        if entry.get("answer_score") is not None:
            continue
        if entry.get("status") in ("error", "missing_image"):
            entry["answer_score"] = "incorrect"
            entry["reasoning_score"] = "missing"
            continue

        if mode == "human":
            result = score_with_human(entry)
            if result is None:
                continue
            answer_score, reasoning_score = result
        else:
            answer_score, reasoning_score = score_with_llm(entry, judge_model)
            print(f"  [{entry['id']}] answer={answer_score} | reasoning={reasoning_score}")

        entry["answer_score"] = answer_score
        entry["reasoning_score"] = reasoning_score

    # Save back
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Summary
    scored = [r for r in results if r.get("answer_score") is not None]
    correct = sum(1 for r in scored if r["answer_score"] == "correct")
    valid_reasoning = sum(1 for r in scored if r["reasoning_score"] == "valid")

    print(f"\n✅ {model_name}")
    print(f"   Correct answers : {correct}/{len(scored)}")
    print(f"   Valid reasoning : {valid_reasoning}/{len(scored)}")
    print(f"   (correct answer + valid reasoning): "
          f"{sum(1 for r in scored if r['answer_score'] == 'correct' and r['reasoning_score'] == 'valid')}/{len(scored)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("--eval", choices=["human", "llm"], required=True)
    parser.add_argument("--judge", default="gpt-4o", help="Judge model (LLM mode only)")
    parser.add_argument("--file", default=None, help="Single result file")
    parser.add_argument("--dir", default=None, help="Directory of result files")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Provide --file or --dir")

    files = []
    if args.file:
        files.append(args.file)
    if args.dir:
        files.extend(str(p) for p in Path(args.dir).glob("*.json"))

    for filepath in sorted(files):
        evaluate_file(filepath, mode=args.eval,
                      judge_model=args.judge if args.eval == "llm" else None)

    print("\n🏁 Evaluation complete. Run leaderboard.py to update the README.")


if __name__ == "__main__":
    main()
