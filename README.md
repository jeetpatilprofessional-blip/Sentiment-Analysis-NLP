# Sentiment Analysis (NLP)

An end-to-end NLP project that explores sentiment/emotion labels in text, preprocesses text, uses TF-IDF features, compares classical machine-learning classifiers, and provides an interactive Streamlit application.

## Demo / Run locally

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Keep `app.py`, `best_sentiment_model.pkl`, and `tfidf_vectorizer.pkl` in the same directory.
4. Launch:
   ```bash
   streamlit run app.py
   ```

## Project workflow

1. Exploratory data analysis of the dataset and sentiment labels.
2. Text preprocessing: lowercase text, remove HTML fragments and URLs, remove digits and non-letter characters, and normalize whitespace.
3. Consolidate infrequent labels (labels with fewer than five examples are grouped as `Other` during the corrected training workflow).
4. Split the dataset into training and test sets before fitting the TF-IDF vectorizer.
5. Represent text using TF-IDF (up to 1,500 features and 1–3 grams in the corrected run).
6. Compare Logistic Regression, Multinomial Naive Bayes, Linear SVC, and Random Forest.
7. Select the model using macro-F1 and evaluate on the held-out test set.
8. Use the saved classifier and vectorizer in the Streamlit inference app.

## Model evaluation\n\nThe original experiment results below are historical and apply to the earlier fine-grained-label setup. Run the updated training script to obtain metrics for the new three-class experiment; do not reuse the old scores as the new model's performance.\n\n## Streamlit app

The app loads the saved model and TF-IDF vectorizer, accepts text input, and displays a predicted label. Where available, it displays the model's top decision scores. **Linear SVC decision scores are ranking signals, not probabilities or confidence percentages.**

## Repository contents

| File | Description |
|---|---|
| `Analysis.ipynb` | Exploratory analysis and model experimentation |
| `sentimentdataset.csv` | Original dataset |
| `cleaned_sentiment_dataset.csv` | Cleaned dataset |
| `train_sentiment.py` | Training, comparison, evaluation, and artifact-saving script |
| `app.py` | Streamlit inference app |
| `best_sentiment_model.pkl` | Saved classifier |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `requirements.txt` | Python dependencies |

## Re-training

Run the training script from the repository directory:

```bash
python train_sentiment.py
```

It maps emotion labels to three broad sentiment classes, removes duplicate text, performs a stratified split before fitting TF-IDF, evaluates a class-balanced Linear SVC, and saves the updated model and vectorizer to the repository directory. Upload the newly generated `.pkl` files to GitHub after running it.

## Limitations and future work

- The dataset is small and labels are highly imbalanced.
- Sentiment and emotion labels can be subjective and context-dependent.
- The preprocessing step removes emojis and punctuation that may convey sentiment.
- Evaluate with additional splits or cross-validation and report per-class metrics.
- Consider a larger dataset, a more clearly defined label taxonomy, and a transformer-based baseline.

Educational portfolio project; not intended for high-stakes decisions.
