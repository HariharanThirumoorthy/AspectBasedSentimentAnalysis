import streamlit as st
import spacy
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import nltk
nltk.download("punkt")

# Load spaCy for aspect extraction
nlp = spacy.load("en_core_web_sm")

# Load ABSA model v3
model_path = "absa_model_v3"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Class label mapping
class_mapping = {0: "Angry", 1: "Happy", 2: "Neutral"}
class_names = ["Angry", "Happy", "Neutral"]

# Predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probs)
    return class_mapping[predicted_class], round(probs[predicted_class] * 100, 2), probs

# Extract aspects
def extract_aspects(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

# LIME wrapper
class SentimentWrapper:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

lime_explainer = LimeTextExplainer(class_names=class_names)
predict_wrapper = SentimentWrapper(tokenizer, model)

# Streamlit UI
st.set_page_config(page_title="ABSA Sentiment Analyzer", layout="centered")
st.title("💬 ABSA Model V3 Sentiment Analyzer")
st.write("Enter a product or service review to analyze sentiment and extract key aspects.")

user_input = st.text_area("📝 Your Review:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip():
        sentiment, confidence, probs = predict_sentiment(user_input)
        aspects = extract_aspects(user_input)

        st.markdown("### ✏️ Your Input:")
        st.write(user_input)

        st.success(f"🗣️ Predicted Sentiment: **{sentiment}** ({confidence}% confidence)")

        if confidence >= 99.99:
            st.warning("⚠️ Very high confidence — double-check if this seems accurate.")

        st.markdown("### 📊 Class Probabilities:")
        for idx, label in class_mapping.items():
            prob = round(probs[idx] * 100, 2)
            bar = "🟩" * int(prob / 5)
            st.write(f"- **{label}**: {prob}% {bar}")

        st.markdown("### 🔍 Extracted Aspects:")
        st.write(aspects if aspects else "(No noun phrases detected)")

        st.markdown("### 🧠 LIME Explanation:")
        exp = lime_explainer.explain_instance(user_input, predict_wrapper, num_features=6)
        st.components.v1.html(exp.as_html(), height=400)

        st.markdown("---")
        st.info("ℹ️ Tip: Misclassified or subtle reviews can help retrain future models.")
    else:
        st.warning("⚠️ Please enter a review first.")
