# 🇦🇿 Azerbaijani Sentiment Analyzer

Binary sentiment classifier fine-tuned on Azerbaijani text using `xlm-roberta-base`.  
Classifies text as **Positive (Müsbət)** or **Negative (Mənfi)**.

---

## 🛠️ Stack

| Tool | Purpose |
|------|---------|
| `xlm-roberta-base` | Base model (multilingual, Azerbaijani-compatible) |
| `Hugging Face Transformers` | Fine-tuning & inference |
| `Streamlit` | Web interface |
| `PyTorch` | Deep learning framework |
| `scikit-learn` | Evaluation metrics |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| F1 (weighted) | ~89% |
| Train epochs | 4 |
| Train samples | ~6100 |

---

## 📁 Dataset

[hajili/azerbaijani_review_sentiment_classification](https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification)  
~6000 balanced real reviews + synthetic augmentation.  
Score ≤ 2 → Negative · Score ≥ 4 → Positive · Score = 3 (neutral) → excluded

---

## 📥 Model Setup

The trained model weights are **not stored in this repo** (too large for git).  
You have two options:

### Option A — Train it yourself

```bash
python train.py
# or with custom args:
python train.py --epochs 4 --batch_size 16 --lr 2e-5 --output_dir ./az-sentiment-model
```

This downloads the dataset from HuggingFace, fine-tunes `xlm-roberta-base`, and saves the model to `./az-sentiment-model/`.

### Option B — Download pre-trained weights from HuggingFace Hub

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Fidan6557/az-sentiment-xlm-roberta"  # replace with actual repo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained("./az-sentiment-model")
model.save_pretrained("./az-sentiment-model")
```

Or update `app.py` to load directly from the HuggingFace Hub instead of a local directory by changing `MODEL_DIR = "./az-sentiment-model"` to the Hub repo name.

---

## 🏃 Run Locally

```bash
# 1. Clone
git clone https://github.com/Fidan6557/az-sentiment-analyzer
cd az-sentiment-analyzer

# 2. Install dependencies  (Python 3.9–3.11 recommended; 3.12+ not tested)
pip install -r requirements.txt

# 3. Obtain the model (see Model Setup above)
python train.py

# 4. Launch the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
az-sentiment-analyzer/
├── app.py               # Streamlit inference UI
├── train.py             # Fine-tuning script
├── requirements.txt
├── .gitignore
├── README.md
└── az-sentiment-model/  # ← created after training (gitignored)
    ├── config.json
    ├── tokenizer files
    └── model weights
```

---

## ⚙️ Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | `xlm-roberta-base` | HuggingFace model name |
| `--dataset` | `hajili/azerbaijani_review_sentiment_classification` | HuggingFace dataset |
| `--output_dir` | `./az-sentiment-model` | Where to save the model |
| `--epochs` | `4` | Number of training epochs |
| `--batch_size` | `16` | Per-device batch size |
| `--lr` | `2e-5` | Learning rate |
| `--max_length` | `128` | Max token length |
| `--seed` | `42` | Random seed |
