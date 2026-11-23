"""Train a 3-class sentiment classifier (Positive / Negative / Neutral)."""
import re, html
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = Path(__file__).parent

POSITIVE = set("""positive happiness joy love amusement enjoyment admiration affection awe acceptance adoration excitement kind pride elation euphoria contentment serenity gratitude hope empowerment compassion tenderness enthusiasm fulfillment reverence anticipation enthusiasm hopeful proud grateful empathetic compassionate playful inspired confident thrill overjoyed inspiration motivation satisfaction blessed appreciation confidence accomplishment wonderment optimism enchantment playfuljoy harmony creativity radiance wonder rejuvenation adventure zest positivity kindness friendship success exploration amazement romance tranquility grandeur energy celebration ecstasy connection touched triumph heartwarming solace breakthrough relief happy""".split())
NEGATIVE = set("""negative anger fear sadness disgust disappointed bitter despair grief loneliness jealousy resentment frustration boredom anxiety intimidation helplessness envy regret numbness melancholy fearful apprehensive overwhelmed jealous devastated frustrated envious dismissive bitterness heartbreak betrayal suffering emotionalstorm isolation disappointment lostlove exhaustion sorrow darkness desperation ruins desolation loss heartache solitude pressure obstacle miscalculation challenge embarrassed sad hate bad""".split())
NEUTRAL = set("""neutral confusion indifference ambivalence contemplation reflection mindfulness pensive curiosity""".split())

def map_label(label):
    label = str(label).strip().lower().replace("-", "").replace(" ", "")
    if label in {x.replace("-", "").replace(" ", "") for x in POSITIVE}: return "Positive"
    if label in {x.replace("-", "").replace(" ", "") for x in NEGATIVE}: return "Negative"
    if label in {x.replace("-", "").replace(" ", "") for x in NEUTRAL}: return "Neutral"
    return None

def clean_text(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)
    text = re.sub(r'[^a-z\\s]', ' ', text)
    return re.sub(r'\\s+', ' ', text).strip()

df = pd.read_csv(BASE / 'sentimentdataset.csv')
df = df[['Text', 'Sentiment']].dropna().copy()
df['Text'] = df['Text'].map(clean_text)
df['Target'] = df['Sentiment'].map(map_label)
df = df[(df.Text != '') & df.Target.notna()].drop_duplicates(subset=['Text']).copy()
print('Rows after cleaning and label mapping:', len(df))
print('Class distribution:\\n', df.Target.value_counts())

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df.Text, df.Target, test_size=0.2, random_state=42, stratify=df.Target
)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy:', round(accuracy_score(y_test, pred), 4))
print('Macro F1:', round(f1_score(y_test, pred, average='macro'), 4))
print(classification_report(y_test, pred, zero_division=0))
joblib.dump(model, BASE/'best_sentiment_model.pkl', compress=3)
joblib.dump(vectorizer, BASE/'tfidf_vectorizer.pkl', compress=3)
print('Saved updated model and vectorizer.')
