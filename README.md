# Aspect-Based Sentiment Analysis (Transformer Fine-Tuning)

The implementation, data layout, configs, and training scripts live in **`absa-sentiment-analysis/`**.

## Quick start

```bash
cd absa-sentiment-analysis
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --model baseline
```

See **`absa-sentiment-analysis/README.md`** for dataset setup, all models, and options (`--rebuild-data`, etc.).
