import re, html
from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE=Path(__file__).parent

def clean_text(text):
    text=html.unescape(str(text)).lower()
    text=re.sub(r'<[^>]+>', ' ', text)
    text=re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text=re.sub(r'\d+', ' ', text)
    text=re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df=pd.read_csv(BASE/'sentimentdataset.csv')
df=df[['Text','Sentiment']].dropna().copy()
df['Text']=df['Text'].map(clean_text)
df['Sentiment']=df['Sentiment'].astype(str).str.strip()
df=df[(df.Text!='')&(df.Sentiment!='')]
counts=df.Sentiment.value_counts()
df['Sentiment']=df.Sentiment.where(df.Sentiment.map(counts)>=5,'Other')
X_train_text,X_test_text,y_train,y_test=train_test_split(df.Text,df.Sentiment,test_size=.2,random_state=42,stratify=df.Sentiment)
vec=TfidfVectorizer(max_features=1500,ngram_range=(1,3),min_df=2,max_df=.95)
X_train=vec.fit_transform(X_train_text); X_test=vec.transform(X_test_text)
models={
 'Logistic Regression': LogisticRegression(max_iter=1000),
 'Multinomial Naive Bayes': MultinomialNB(),
 'Linear SVC': LinearSVC(),
 'Random Forest': RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced_subsample',n_jobs=-1),
}
results=[]
for name,m in models.items():
 m.fit(X_train,y_train); pred=m.predict(X_test)
 results.append((name,accuracy_score(y_test,pred),f1_score(y_test,pred,average='macro',zero_division=0),f1_score(y_test,pred,average='weighted',zero_division=0)))
print('RESULTS')
for r in results: print(r)
best_name,best_acc,best_macro,best_weighted=max(results,key=lambda x:x[2])
best=models[best_name]
print('\nBEST BY MACRO F1:',best_name, best_acc,best_macro,best_weighted)
print(classification_report(y_test,best.predict(X_test),zero_division=0))
joblib.dump(best,BASE/'best_sentiment_model.pkl',compress=3)
joblib.dump(vec,BASE/'tfidf_vectorizer.pkl',compress=3)
print('Saved artifacts')
