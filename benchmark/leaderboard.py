"""
leaderboard.py — Generate leaderboard and update README.md

Usage:
    python benchmark/leaderboard.py
    python benchmark/leaderboard.py --dir results/2026-06-30/
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DIFFICULTIES = ["95%", "50%", "10%", "1%"]


README_TEMPLATE = """# 🧠 llm-logic-bench

> A public benchmark comparing LLMs on **visual logic puzzles** inspired by the *100% Logique* TV show.
> All questions are **image-based** — custom-made to avoid training data contamination.
> Difficulty levels: **95%** · **50%** · **10%** · **1%** (% of humans who answer correctly)

---

## 🏆 Leaderboard

> Scoring: **1pt** for correct answer with valid reasoning · **0.5pt** for correct answer with flawed reasoning · **0pt** otherwise
> Token counts include the image encoding overhead.

{leaderboard_table}

> Last updated: **{date}** | Dataset: **{dataset_version}** | Evaluation: **{eval_mode}**

---

## 📊 Breakdown by Difficulty

{breakdown_table}

---

## 🔍 Answer vs Reasoning

{reasoning_table}

---

## ⚡ Token Efficiency

> Average tokens consumed **per question** (input = prompt + image encoding, output = model response).

{token_table}

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/your-username/llm-logic-bench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API keys
cp .env.example .env
# Edit .env with your own keys

# 4. (Optional) Use your own dataset
# Replace questions/v1/dataset.json + add images to questions/v1/images/

# 5. Run the benchmark
python benchmark/runner.py

# Test on fewer questions first:
python benchmark/runner.py --limit 3

# 6. Evaluate results
python benchmark/evaluator.py --eval human --dir results/YYYY-MM-DD/
# or:
python benchmark/evaluator.py --eval llm --judge gpt-4o --dir results/YYYY-MM-DD/

# 7. Update leaderboard
python benchmark/leaderboard.py
```

---

## 📁 Dataset Format

```json
{{
  "id": "q_001",
  "difficulty": "50%",
  "category": "visual sequence",
  "question": "What element replaces the question mark?",
  "image": "images/q_001.jpg",
  "answer": "A black circle."
}}
```

| Field | Values | Description |
|---|---|---|
| `difficulty` | `95%` `50%` `10%` `1%` | % of humans who answer correctly |
| `image` | path | Relative path to image (always required) |
| `answer` | string | The single correct answer |

---

## 📜 License

MIT — Fork it, add your own images and questions, run your own leaderboard.
"""


def load_results(results_dir: str) -> list[dict]:
    all_data = []
    for filepath in sorted(Path(results_dir).glob("*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            all_data.append(json.load(f))
    return all_data


def compute_stats(data: dict) -> dict:
    all_results = data["results"]
    scored = [r for r in all_results if r.get("answer_score") is not None]

    stats = {
        "model": data["meta"]["model"],
        "total": len(scored),
        "by_difficulty": defaultdict(lambda: {"total": 0, "full": 0, "half": 0}),
        "correct_answers": 0,
        "valid_reasoning": 0,
        "full_score": 0,  # correct answer + valid reasoning = 1pt
        "half_score": 0,  # correct answer + flawed reasoning = 0.5pt
        # Token stats (computed over ALL results, scored or not)
        "tokens": {
            "avg_input":  round(sum(r["tokens"]["input"]  for r in all_results) / max(len(all_results), 1)),
            "avg_output": round(sum(r["tokens"]["output"] for r in all_results) / max(len(all_results), 1)),
            "avg_total":  round(sum(r["tokens"]["total"]  for r in all_results) / max(len(all_results), 1)),
            "grand_total": data["meta"].get("total_tokens", {}).get("total", 0),
        },
    }

    for r in scored:
        diff = r.get("difficulty", "unknown")
        ans = r.get("answer_score", "incorrect")
        rea = r.get("reasoning_score", "missing")

        stats["by_difficulty"][diff]["total"] += 1

        if ans == "correct":
            stats["correct_answers"] += 1
            if rea == "valid":
                stats["full_score"] += 1
                stats["by_difficulty"][diff]["full"] += 1
                stats["valid_reasoning"] += 1
            else:
                stats["half_score"] += 1
                stats["by_difficulty"][diff]["half"] += 1
        elif rea == "valid":
            stats["valid_reasoning"] += 1

    return stats


def fmt(n: float, total: int, suffix: str = "") -> str:
    if total == 0:
        return "—"
    pct = int(n / total * 100)
    return f"{n}/{total} ({pct}%){suffix}"


def build_leaderboard_table(all_stats: list[dict]) -> str:
    # Sort by full_score desc, then half_score desc
    all_stats.sort(key=lambda s: (s["full_score"] + 0.5 * s["half_score"]), reverse=True)

    header = "| Rank | Model | Score | Correct answers | Valid reasoning |"
    sep    = "|------|-------|-------|-----------------|-----------------|"
    rows = [header, sep]

    for i, s in enumerate(all_stats, 1):
        total_score = s["full_score"] + 0.5 * s["half_score"]
        max_score = s["total"]
        score_pct = int(total_score / max_score * 100) if max_score else 0
        rows.append(
            f"| {i} | {s['model']} | **{total_score:.1f}/{max_score} ({score_pct}%)** "
            f"| {fmt(s['correct_answers'], s['total'])} "
            f"| {fmt(s['valid_reasoning'], s['total'])} |"
        )

    return "\n".join(rows)


def build_breakdown_table(all_stats: list[dict]) -> str:
    header = "| Model | " + " | ".join(DIFFICULTIES) + " |"
    sep    = "|-------|" + "-------|" * len(DIFFICULTIES)
    rows = [header, sep]

    for s in all_stats:
        cols = [s["model"]]
        for diff in DIFFICULTIES:
            d = s["by_difficulty"].get(diff, {"total": 0, "full": 0, "half": 0})
            score = d["full"] + 0.5 * d["half"]
            cols.append(fmt(score, d["total"]))
        rows.append("| " + " | ".join(cols) + " |")

    return "\n".join(rows)


def build_reasoning_table(all_stats: list[dict]) -> str:
    header = "| Model | Correct answer | Valid reasoning | Both correct |"
    sep    = "|-------|----------------|-----------------|--------------|"
    rows = [header, sep]

    for s in all_stats:
        rows.append(
            f"| {s['model']} "
            f"| {fmt(s['correct_answers'], s['total'])} "
            f"| {fmt(s['valid_reasoning'], s['total'])} "
            f"| {fmt(s['full_score'], s['total'])} |"
        )

    return "\n".join(rows)


def build_token_table(all_stats: list[dict]) -> str:
    header = "| Model | Avg input tokens | Avg output tokens | Avg total / question | Grand total |"
    sep    = "|-------|-----------------|-------------------|----------------------|-------------|"
    rows = [header, sep]

    for s in all_stats:
        t = s["tokens"]
        rows.append(
            f"| {s['model']} "
            f"| {t['avg_input']:,} "
            f"| {t['avg_output']:,} "
            f"| **{t['avg_total']:,}** "
            f"| {t['grand_total']:,} |"
        )

    return "\n".join(rows)


def update_readme(leaderboard_table, breakdown_table, reasoning_table, token_table,
                  dataset_version, eval_mode):
    content = README_TEMPLATE.format(
        leaderboard_table=leaderboard_table,
        breakdown_table=breakdown_table,
        reasoning_table=reasoning_table,
        token_table=token_table,
        date=datetime.now().strftime("%Y-%m-%d"),
        dataset_version=dataset_version,
        eval_mode=eval_mode,
    )
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md updated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="Results directory")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--eval-mode", default="human")
    args = parser.parse_args()

    results_base = Path("results")
    if args.dir:
        results_dir = args.dir
    else:
        subdirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
        if not subdirs:
            print("❌ No result directories found in results/")
            return
        results_dir = str(subdirs[-1])

    print(f"📂 Loading results from: {results_dir}")
    all_data = load_results(results_dir)

    if not all_data:
        print("❌ No result files found.")
        return

    all_stats = [compute_stats(d) for d in all_data]

    leaderboard_table = build_leaderboard_table(all_stats)
    breakdown_table   = build_breakdown_table(all_stats)
    reasoning_table   = build_reasoning_table(all_stats)
    token_table       = build_token_table(all_stats)

    print("\n" + leaderboard_table)
    update_readme(leaderboard_table, breakdown_table, reasoning_table, token_table,
                  args.dataset_version, args.eval_mode)



if __name__ == "__main__":
    main()
