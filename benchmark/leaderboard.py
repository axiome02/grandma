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
SCORE_WEIGHTS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

README_TEMPLATE = """# 🧠 llm-logic-bench

> A public benchmark comparing LLMs on logic puzzles inspired by the *100% Logique* TV show.
> Questions span 4 difficulty levels: **95%**, **50%**, **10%**, **1%** (% of humans who answer correctly).
> Dataset: English | Question types: Text & Multimodal (text + image)

---

## 🏆 Leaderboard

{leaderboard_table}

> Last updated: **{date}** | Dataset version: **{dataset_version}** | Evaluation: **{eval_mode}**

---

## 📊 Breakdown by Difficulty

{breakdown_table}

---

## 🖼️ Text vs Multimodal

{multimodal_table}

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

# 4. Run the benchmark
python benchmark/runner.py

# 5. Evaluate results
python benchmark/evaluator.py --eval human
# or
python benchmark/evaluator.py --eval llm --judge gpt-4o

# 6. Update leaderboard
python benchmark/leaderboard.py
```

> You can also bring your own dataset — just replace `questions/v1/dataset.json`.

---

## 📁 Dataset

Questions are stored in `questions/v1/dataset.json`.
Each question has a unique id, difficulty level, type (text/multimodal), and a single correct answer.

| Field | Description |
|---|---|
| `id` | Unique identifier (e.g. `q_001`) |
| `type` | `text` or `multimodal` |
| `difficulty` | `95%`, `50%`, `10%`, or `1%` |
| `category` | Logic sub-category |
| `question` | Question in English |
| `image` | Path to image (null for text-only) |
| `answer` | The one and only correct answer |

---

## 📜 License

MIT — Feel free to fork, contribute questions, or run your own evaluation.
"""


def load_results(results_dir: str) -> list[dict]:
    all_data = []
    for filepath in sorted(Path(results_dir).glob("*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_data.append(data)
    return all_data


def compute_stats(data: dict) -> dict:
    results = [r for r in data["results"] if r.get("score") is not None]

    stats = {
        "model": data["meta"]["model"],
        "total": len(results),
        "by_difficulty": defaultdict(lambda: {"total": 0, "score": 0.0}),
        "by_type": defaultdict(lambda: {"total": 0, "score": 0.0}),
        "global_score": 0.0,
    }

    for r in results:
        weight = SCORE_WEIGHTS.get(r["score"], 0.0)
        diff = r.get("difficulty", "unknown")
        qtype = r.get("type", "text")

        stats["by_difficulty"][diff]["total"] += 1
        stats["by_difficulty"][diff]["score"] += weight
        stats["by_type"][qtype]["total"] += 1
        stats["by_type"][qtype]["score"] += weight
        stats["global_score"] += weight

    return stats


def format_score(score: float, total: int) -> str:
    if total == 0:
        return "—"
    pct = int(score / total * 100)
    return f"{int(score)}/{total} ({pct}%)"


def build_leaderboard_table(all_stats: list[dict]) -> str:
    all_stats.sort(key=lambda s: s["global_score"], reverse=True)

    header = "| Rank | Model | 95% | 50% | 10% | 1% | **Global** |"
    sep = "|------|-------|-----|-----|-----|----|------------|"
    rows = [header, sep]

    for i, stats in enumerate(all_stats, 1):
        cols = [f"{i}", stats["model"]]
        for diff in DIFFICULTIES:
            d = stats["by_difficulty"].get(diff, {"total": 0, "score": 0.0})
            cols.append(format_score(d["score"], d["total"]))
        cols.append(f"**{format_score(stats['global_score'], stats['total'])}**")
        rows.append("| " + " | ".join(cols) + " |")

    return "\n".join(rows)


def build_breakdown_table(all_stats: list[dict]) -> str:
    header = "| Model | " + " | ".join(DIFFICULTIES) + " |"
    sep = "|-------|" + "------|" * len(DIFFICULTIES)
    rows = [header, sep]

    for stats in all_stats:
        cols = [stats["model"]]
        for diff in DIFFICULTIES:
            d = stats["by_difficulty"].get(diff, {"total": 0, "score": 0.0})
            cols.append(format_score(d["score"], d["total"]))
        rows.append("| " + " | ".join(cols) + " |")

    return "\n".join(rows)


def build_multimodal_table(all_stats: list[dict]) -> str:
    header = "| Model | Text | Multimodal |"
    sep = "|-------|------|------------|"
    rows = [header, sep]

    for stats in all_stats:
        text = stats["by_type"].get("text", {"total": 0, "score": 0.0})
        multi = stats["by_type"].get("multimodal", {"total": 0, "score": 0.0})
        rows.append(
            f"| {stats['model']} | {format_score(text['score'], text['total'])} | {format_score(multi['score'], multi['total'])} |"
        )

    return "\n".join(rows)


def update_readme(leaderboard_table: str, breakdown_table: str, multimodal_table: str,
                  dataset_version: str, eval_mode: str):
    content = README_TEMPLATE.format(
        leaderboard_table=leaderboard_table,
        breakdown_table=breakdown_table,
        multimodal_table=multimodal_table,
        date=datetime.now().strftime("%Y-%m-%d"),
        dataset_version=dataset_version,
        eval_mode=eval_mode,
    )
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md updated.")


def main():
    parser = argparse.ArgumentParser(description="Generate leaderboard from results")
    parser.add_argument("--dir", default=None, help="Results directory (default: latest in results/)")
    parser.add_argument("--dataset-version", default="v1", help="Dataset version label")
    parser.add_argument("--eval-mode", default="human", help="Evaluation mode used (human or llm)")
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
    breakdown_table = build_breakdown_table(all_stats)
    multimodal_table = build_multimodal_table(all_stats)

    print("\n" + leaderboard_table)
    update_readme(leaderboard_table, breakdown_table, multimodal_table, args.dataset_version, args.eval_mode)


if __name__ == "__main__":
    main()
