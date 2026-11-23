# Sentiment Analysis (NLP)

An end-to-end Natural Language Processing project that explores text and sentiment labels, preprocesses text, represents it with TF-IDF features, compares classical machine-learning classifiers, and provides an interactive Streamlit demo.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Keep `app.py`, `best_sentiment_model.pkl`, and `tfidf_vectorizer.pkl` in the same directory.

## Workflow
1. Exploratory analysis of dataset dimensions, missing values, label frequencies, and metadata.
2. Text preprocessing: lowercase, remove HTML fragments, URLs, digits, punctuation/symbols, and normalize whitespace.
3. Label handling: the dataset contains many distinct labels, including infrequent classes. Grouping rare labels should be documented and applied consistently.
4. TF-IDF representation with n-gram features (configuration depends on the trained artifact).
5. Compare Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest.
6. Evaluate on a held-out set using accuracy and per-class metrics; due to fragmented labels, accuracy alone is insufficient.
7. Interactive inference with Streamlit.

## Files
| File | Description |
|---|---|
| `Analysis.ipynb` | EDA, preprocessing, model experiments, evaluation |
| `sentimentdataset.csv` | Original dataset |
| `cleaned_sentiment_dataset.csv` | Dataset including preprocessing output |
| `app.py` | Streamlit inference application |
| `best_sentiment_model.pkl` | Serialized classifier |
| `tfidf_vectorizer.pkl` | Serialized TF-IDF vectorizer |
| `requirements.txt` | Dependencies |

## Dataset and limitations
The dataset contains 732 rows and a large number of sentiment classes. Labels are unevenly represented, making this a fine-grained emotion/sentiment classification task rather than a simple binary classification problem. Small sample sizes, rare classes, subjective labels, and text cleaning that removes emojis/punctuation can affect generalization.

## Reproducibility note
The original experiment fitted TF-IDF on the complete dataset before splitting, which can leak test-set information into feature selection. For a rigorous re-evaluation, split raw texts first and fit the vectorizer only on training texts (or use a scikit-learn Pipeline). The existing model artifact has not been retrained unless the corrected workflow is run.

## Next steps
- Re-evaluate with a leakage-free pipeline.
- Report macro-F1, per-class precision/recall, and confusion matrix.
- Consider a documented label taxonomy and larger dataset.
- Compare a transformer baseline.

Educational portfolio project; not intended for high-stakes decisions.
