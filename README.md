# 🧠 llm-logic-bench

> A public benchmark comparing LLMs on logic puzzles inspired by the *100% Logique* TV show.
> Questions span 4 difficulty levels: **95%**, **50%**, **10%**, **1%** (% of humans who answer correctly).
> Dataset: English | Question types: Text & Multimodal (text + image)

---

## 🏆 Leaderboard

*Run the benchmark and generate results to populate this table.*

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
# Replace questions/v1/dataset.json with your own questions

# 5. Configure which models to run
# Edit config.yaml — set enabled: true/false per model

# 6. Run the benchmark
python benchmark/runner.py

# To test with only 5 questions first:
python benchmark/runner.py --limit 5

# 7. Evaluate results
python benchmark/evaluator.py --eval human --dir results/YYYY-MM-DD/
# or use LLM-as-a-judge:
python benchmark/evaluator.py --eval llm --judge gpt-4o --dir results/YYYY-MM-DD/

# 8. Update leaderboard
python benchmark/leaderboard.py
```

---

## 📁 Dataset Format

Questions are stored in `questions/v1/dataset.json`.

```json
{
  "id": "q_001",
  "type": "text",
  "difficulty": "95%",
  "category": "logical deduction",
  "question": "...",
  "image": null,
  "answer": "..."
}
```

| Field | Values | Description |
|---|---|---|
| `type` | `text` or `multimodal` | Text only or text + image |
| `difficulty` | `95%` `50%` `10%` `1%` | % of humans who answer correctly |
| `image` | path or `null` | Relative path to image file |
| `answer` | string | The single correct answer |

---

## 🔧 Configuration (`config.yaml`)

Enable or disable models and choose your evaluation mode:

```yaml
models:
  - name: gpt-4o
    provider: openai
    enabled: true

evaluation:
  mode: human   # human | llm
  judge_model: gpt-4o
```

---

## 📜 License

MIT — Fork it, add your own questions, run your own leaderboard.
