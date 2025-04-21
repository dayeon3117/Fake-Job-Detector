"""
Deep learning model using DistilBERT trained on a balanced dataset.
Upsamples fake job postings and trains using Hugging Face Trainer.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

class FakeJobDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_data(tokenizer, texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True)
    return FakeJobDataset(encodings, labels.tolist())

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("Deep Model Evaluation (Balanced):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def balance_dataset(df):
    df_majority = df[df['fraudulent'] == 0]
    df_minority = df[df['fraudulent'] == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

if __name__ == "__main__":
    df = pd.read_csv("data/raw/fake_job_postings.csv")
    df = df[['description', 'fraudulent']].dropna().reset_index(drop=True)
    df_balanced = balance_dataset(df)

    df_train, df_test = train_test_split(df_balanced, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = tokenize_data(tokenizer, df_train["description"], df_train["fraudulent"])
    test_dataset = tokenize_data(tokenizer, df_test["description"], df_test["fraudulent"])

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./models/distilbert_balanced",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_dir="./logs_balanced",
        seed=42,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)

    df_out = pd.DataFrame({
        "actual": df_test["fraudulent"].values,
        "predicted": pred_labels
    })
    df_out.to_csv("data/outputs/deep_predictions_balanced.csv", index=False)

    model.save_pretrained("./models/distilbert_balanced")
    tokenizer.save_pretrained("./models/distilbert_balanced")
