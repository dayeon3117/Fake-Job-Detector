"""
Evaluates the performance of the deep model trained on the balanced dataset.
Uses the final checkpoint to predict and print performance metrics.
"""
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report

# Load test data
df = pd.read_csv("data/raw/fake_job_postings.csv")
df = df[['description', 'fraudulent']].dropna().reset_index(drop=True)
df_test = df.sample(frac=0.2, random_state=42)  # use same seed

# Load balanced model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_balanced/checkpoint-4500")
model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_balanced/checkpoint-4500")
model.eval()

# Tokenize and predict
preds = []
for text in df_test["description"]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

# Evaluation
print("Deep Model Evaluation (Balanced):")
print(classification_report(df_test["fraudulent"].values, preds, digits=4))

# Save predictions
df_out = pd.DataFrame({
    "actual": df_test["fraudulent"].values,
    "predicted": preds
})
df_out.to_csv("data/outputs/deep_predictions_balanced.csv", index=False)

