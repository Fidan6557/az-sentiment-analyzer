"""
app.py — Streamlit inference UI for the Azerbaijani Sentiment Analyzer.

Run:
    streamlit run app.py

The model directory (./az-sentiment-model) must exist locally.
See README.md → "Model Setup" for how to obtain it.
"""

import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AZ Sentiment Analyzer",
    page_icon="🇦🇿",
    layout="centered",
)

st.title("🇦🇿 Azərbaycan Sentimenti Analizatoru")
st.markdown(
    "Azərbaycan dilində mətn daxil edin — model müsbət və ya mənfi olduğunu müəyyən edəcək."
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_DIR = "./az-sentiment-model"

LABEL_MAP = {
    0: ("Mənfi", "😞", "#FF4B4B"),
    1: ("Müsbət", "😊", "#00C851"),
}

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Model yüklənir, zəhmət olmasa gözləyin...")
def load_model():
    if not os.path.isdir(MODEL_DIR):
        st.error(
            f"Model qovluğu tapılmadı: `{MODEL_DIR}`\n\n"
            "Zəhmət olmasa README.md → **Model Setup** bölməsini oxuyun "
            "və modeli endirin."
        )
        st.stop()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────

def predict(text: str):
    """Return (label_index, probs_tensor) for the given input text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    return probs.argmax().item(), probs


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

text_input = st.text_area(
    "Mətn daxil edin:",
    placeholder="Məsələn: Azərbaycan iqtisadiyyatı sürətlə inkişaf edir...",
    height=140,
)

if st.button("Analiz et 🔍", use_container_width=True):
    if not text_input.strip():
        st.warning("Zəhmət olmasa mətn daxil edin.")
    else:
        with st.spinner("Analiz edilir..."):
            pred_idx, probs = predict(text_input.strip())

        # Defensive check — guard against unexpected model output
        if pred_idx not in LABEL_MAP:
            st.error(
                f"Gözlənilməyən model çıxışı: class index = {pred_idx}. "
                "Model düzgün fine-tune edilib? LABEL_MAP-ı yoxlayın."
            )
            st.stop()

        label_name, emoji, color = LABEL_MAP[pred_idx]
        confidence = probs[pred_idx].item()

        st.markdown("---")
        st.markdown(
            f"<h3 style='color:{color}'>{emoji} Nəticə: {label_name}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Etibar:** `{confidence:.1%}`")

        st.markdown("**Bütün ehtimallar:**")
        for idx, (name, em, col) in LABEL_MAP.items():
            prob_val = probs[idx].item()
            st.markdown(
                f"<span style='color:{col}'>{em} **{name}**</span>: `{prob_val:.1%}`",
                unsafe_allow_html=True,
            )
            st.progress(float(prob_val))

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "Made with ❤️ | Model: `xlm-roberta-base` fine-tuned on Azerbaijani text",
    unsafe_allow_html=False,
)
