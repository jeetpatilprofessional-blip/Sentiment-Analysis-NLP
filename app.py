import re
import joblib
import streamlit as st


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model, vectorizer = load_artifacts()

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")
st.title("Sentiment Analysis")
st.write("Enter text to predict its sentiment.")

text = st.text_area("Text", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(text)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        st.subheader("Prediction")
        st.success(str(prediction).strip())
