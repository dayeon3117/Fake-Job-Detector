"""
Improved classical model using Random Forest Classifier.
Balanced training data + n-grams + more features.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def balance_dataset(df):
    majority = df[df['fraudulent'] == 0]
    minority = df[df['fraudulent'] == 1]
    upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    return pd.concat([majority, upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def train_rf_model(df):
    X = df['description']
    y = df['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Random Forest Model Evaluation:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    joblib.dump(model, "models/classical_model_rf.pkl")
    joblib.dump(vectorizer, "models/vectorizer_rf.pkl")
    return pd.DataFrame({'actual': y_test, 'predicted': y_pred})

def load_rf_components():
    model = joblib.load("models/classical_model_rf.pkl")
    vectorizer = joblib.load("models/vectorizer_rf.pkl")
    return model, vectorizer

def get_rf_prediction(text: str):
    model, vectorizer = load_rf_components()
    features = vectorizer.transform([text])
    pred = model.predict(features)[0]
    return "Suspicious" if pred == 1 else "Real"

if __name__ == "__main__":
    df = pd.read_csv("data/raw/fake_job_postings.csv")
    df = df[['description', 'fraudulent']].dropna()
    df_balanced = balance_dataset(df)
    print(df_balanced['fraudulent'].value_counts())
    results = train_rf_model(df_balanced)
    results.to_csv("data/outputs/classical_predictions_rf.csv", index=False)
