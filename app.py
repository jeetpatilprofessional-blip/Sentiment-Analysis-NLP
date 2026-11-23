import re
from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
MODEL_PATH = Path("best_sentiment_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model/vectorizer files are missing. Place both .pkl files beside app.py.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("💬 Sentiment Analysis")
st.caption("Classify text using a TF-IDF representation and a trained machine-learning model.")

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

text_input = st.text_area("Enter text to analyze", placeholder="Type a sentence, review, or social-media post…", height=160)
if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter text before analyzing.")
    else:
        cleaned = clean_text(text_input)
        if not cleaned:
            st.warning("The text contains no usable words after preprocessing.")
        else:
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            st.subheader("Prediction")
            st.success(str(prediction))
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)[:5]
                st.subheader("Top class probabilities")
                st.bar_chart({label: float(prob) for label, prob in ranked})
            st.caption("This prediction reflects the training labels and limitations of the dataset.")

with st.expander("About this project"):
    st.write("This demo uses text preprocessing, TF-IDF features, and a machine-learning classifier. Predictions may not capture nuanced or context-dependent sentiment.")
