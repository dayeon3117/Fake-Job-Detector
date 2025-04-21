"""
Compares the performance of the naive, classical and deep models.
Prints classification reports for each model.
"""

import pandas as pd
from sklearn.metrics import classification_report

# Load predictions
deep = pd.read_csv("data/outputs/deep_predictions_balanced.csv")
rf = pd.read_csv("data/outputs/classical_predictions_rf.csv")
naive = pd.read_csv("data/outputs/naive_predictions.csv")

print("=== Deep Model (DistilBERT) ===")
print(classification_report(deep["actual"], deep["predicted"], digits=4))

print("\n=== Classical Model (Random Forest) ===")
print(classification_report(rf["actual"], rf["predicted"], digits=4))

print("\n=== Naive Model ===")
print(classification_report(naive["fraudulent"], naive["prediction"], digits=4))
