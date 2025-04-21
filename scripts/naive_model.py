"""
Naive model that predicts the majority class for all job postings.
Includes evaluation and a simple prediction function used by the app.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def load_data(path):
    df = pd.read_csv(path)
    df = df[['description', 'fraudulent']].dropna()
    return df

def evaluate_naive_model(df):
    majority_class = df['fraudulent'].mode()[0]
    df['prediction'] = majority_class

    acc = accuracy_score(df['fraudulent'], df['prediction'])
    prec = precision_score(df['fraudulent'], df['prediction'], zero_division=0)
    rec = recall_score(df['fraudulent'], df['prediction'], zero_division=0)
    f1 = f1_score(df['fraudulent'], df['prediction'], zero_division=0)

    print("Naive Model Evaluation:")
    print(f"Majority class predicted: {majority_class}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    joblib.dump(majority_class, "models/naive_majority_class.pkl")
    return df[['fraudulent', 'prediction']]

def load_majority_class():
    return joblib.load("models/naive_majority_class.pkl")

def get_naive_prediction(text: str):
    majority_class = load_majority_class()
    return "Suspicious" if majority_class == 1 else "Real"

if __name__ == "__main__":
    df = load_data("data/raw/fake_job_postings.csv")
    results = evaluate_naive_model(df)
    results.to_csv("data/outputs/naive_predictions.csv", index=False)
