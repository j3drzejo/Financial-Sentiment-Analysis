# 📊 Financial Sentiment Analysis

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-FinBERT-FFD21E)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](projekt.ipynb)

**Three-class sentiment classification of financial texts using classical NLP models and FinBERT fine-tuning**

</div>

---

## Overview

This project investigates how different design choices affect sentiment classification quality on financial texts. The dataset combines formal earnings reports with informal investment tweets, creating interesting challenges for NLP models — domain-specific jargon, class imbalance, and subtle negative language expressed through financial euphemisms.

**Research questions explored:**

| # | Question | Finding |
|---|----------|---------|
| 1 | Do bigrams improve results over unigrams? | ❌ They hurt — dataset too small for ~90k bigram features |
| 2 | Does TF-IDF outperform Bag-of-Words? | ✅ Yes — especially on the *negative* class |
| 3 | Does standard preprocessing help in the financial domain? | ❌ It hurts — destroys decimal numbers and compound words |
| 4 | Does fine-tuning FinBERT significantly improve results? | ✅ Clearly — +12 pp. macro F1 over the best classical model |

---

## Results

| Model | Accuracy | Balanced Acc. | Macro F1 |
|-------|:--------:|:-------------:|:--------:|
| Baseline A — BoW unigrams | 0.6775 | 0.5956 | 0.5978 |
| Baseline B — TF-IDF unigrams | 0.6946 | 0.6488 | 0.6400 |
| Exp. 1A — BoW + bigrams | 0.6801 | 0.5814 | 0.5854 |
| Exp. 1B — TF-IDF + bigrams | 0.6878 | 0.6328 | 0.6280 |
| Exp. 2 — TF-IDF + bigrams + preprocessing | 0.6826 | 0.6217 | 0.6179 |
| **Exp. 3 — FinBERT (fine-tuned)** | **0.8007** | **0.7634** | **0.7501** |

> Primary metrics are **balanced accuracy** and **macro F1** due to class imbalance. A naive majority-class classifier would achieve ~54% accuracy while learning nothing.

---

## Dataset

**[Financial Sentiment Analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)** (Kaggle, sbhatti) — a combination of two public sources:

- **Financial PhraseBank** — sentences from earnings reports, manually annotated by domain experts (Malo et al., 2014)
- **FiQA** — tweets and questions from investment forums

5,842 examples · split 70 / 10 / 20 (train / val / test) with stratification · labels: `positive` (32%), `neutral` (54%), `negative` (15%)

> The dataset is not included in this repository due to Kaggle's terms of service. Download `data.csv` manually from the link above and place it in the project root before running the notebook.

---

## Getting Started

### Prerequisites

- Python 3.10+
- ~5 GB of free disk space (PyTorch + FinBERT weights)

### 1 — Clone the repository

```bash
git clone https://github.com/j3drzejo/Financial-Sentiment-Analysis.git
cd Financial-Sentiment-Analysis
```

### 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Download NLTK resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 5 — Download the dataset

Go to [kaggle.com/datasets/sbhatti/financial-sentiment-analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis), download the CSV and save it as `data.csv` in the project root.

### 6 — Launch the notebook

```bash
jupyter notebook projekt.ipynb
```

---

## Hardware

FinBERT fine-tuning was run on Apple Silicon (MPS). Expected training times for 3 epochs:

| Hardware | Time |
|----------|------|
| Apple Silicon (MPS) | ~20 min |
| NVIDIA GPU (CUDA) | ~10–15 min |
| CPU only | ~60–90 min |

The training cell automatically uses MPS if available. Change `device = 'mps'` to `'cuda'` or `'cpu'` as needed.

---

## Project Structure

```
Financial-Sentiment-Analysis/
├── projekt.ipynb        # Main notebook — full code and analysis
├── data.csv             # Dataset (download manually, see above)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Dependencies

| Package | Used for |
|---------|----------|
| `scikit-learn` | Logistic Regression, CountVectorizer, TfidfVectorizer, metrics |
| `nltk` | Stop-word lists, tokenisation |
| `transformers` | FinBERT model, Trainer API |
| `datasets` | DataFrame → Hugging Face Dataset conversion |
| `torch` | Deep learning backend |
| `pandas` / `numpy` | Data processing |
| `matplotlib` | Visualisations |

---

## License

[MIT](LICENSE)
