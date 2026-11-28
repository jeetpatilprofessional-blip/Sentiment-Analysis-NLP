# Sentiment Analysis | NLP with Machine Learning

An interactive Natural Language Processing (NLP) application that accepts text, converts it into TF-IDF features, and uses a trained machine-learning classifier to produce a label prediction.

## Application Preview

![Sentiment Analysis Streamlit Application](Sentiment%20Analysis%20Interface.webp)

The screenshot shows the Streamlit interface with a text input area, an **Analyze sentiment** button, and a displayed prediction.

## Features

- Interactive text input through a Streamlit web interface.
- Text cleaning before inference, including lowercasing and removal of HTML fragments, URLs, digits, punctuation, and extra whitespace.
- TF-IDF vectorization to represent text numerically.
- Prediction using a saved machine-learning classifier.
- Displays the predicted label.
- Displays top model decision scores when the classifier supports `decision_function`.

> **Important:** Decision scores are ranking signals, not probabilities or confidence percentages. The displayed prediction is based on the current saved model and its training labels. It should not be treated as a definitive interpretation of a person's feelings or opinions.

## Technology Stack

- Python
- Natural Language Processing (NLP)
- Pandas
- Scikit-learn
- TF-IDF
- Linear Support Vector Classification (Linear SVC)
- Streamlit
- Joblib
- Jupyter Notebook

## Project Workflow

1. Explore the sentiment dataset and its labels.
2. Clean and normalize text.
3. Convert text into numerical TF-IDF features.
4. Train a machine-learning classifier.
5. Save the trained classifier and TF-IDF vectorizer.
6. Load the saved artifacts in the Streamlit application.
7. Process user-entered text and display the predicted label and, where supported, model decision scores.

## Repository Structure

```
Sentiment-Analysis-NLP/
├── Analysis.ipynb
├── app.py
├── train_sentiment.py
├── sentimentdataset.csv
├── cleaned_sentiment_dataset.csv
├── best_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── Sentiment Analysis Interface.webp
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/jeetpatilprofessional-blip/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP
```

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The application should open in your browser.

## Model and Training Notes

- `app.py` loads `best_sentiment_model.pkl` and `tfidf_vectorizer.pkl` and uses them for predictions.
- `train_sentiment.py` contains a training workflow intended to map dataset emotion labels into Positive, Negative, and Neutral classes and train a Linear SVC model.
- The currently committed model artifact and the training script may not represent the same training run. The application screenshot reflects the output of the currently committed model artifact. Retrain the model and replace both saved `.pkl` files together before claiming that the three-class training workflow is the model currently used by the app.
- No accuracy or F1 score is advertised here; use verified output from the same model artifact before reporting evaluation metrics.

## Limitations

- Sentiment is context-dependent, and short text, sarcasm, slang, or mixed opinions can be difficult to classify.
- The dataset's label quality and class distribution affect model behavior.
- Text preprocessing removes punctuation and other signals that may carry meaning.
- Predictions and decision scores are model outputs, not calibrated probabilities or guaranteed interpretations.

Educational portfolio project; not intended for high-stakes decisions.
