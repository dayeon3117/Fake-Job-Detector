# Fake-Job-Detector

Fake job listings are becoming more common and can lead to serious problems like identity theft or money loss. Job platforms like LinkedIn and Indeed use their own tools to catch scams, but these tools are not available to public.

This project creates a public tool that lets anyone check if a job post seems suspicious. It uses different models to make predictions and runs through a simple web app that anyone can use.

## Problem Statement

Scam job listings are hard to spot, especially for people applying to jobs for the first time. This tool gives users a quick way to check a job description to see if it might be fake. It predicts if a listing is real or suspicious based on how it is written.

## Dataset

I used the [Fake Job Posting Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) from Kaggle. The dataset has job titles, descriptions and labels showing whether a job is real or fake.

The dataset is highly imbalanced; only about 5% of the job postings are labeled as fake. To deal with this, I balanced the training data using upsampling before training the classical and deep learning models.

## Previous Approaches

Fake job posts have been around for a long time. Most earlier work used keyword filters or classical models like Naive Bayes or logistic regression. These models looked for red flags like vague descriptions or missing contact details. They worked for some simple scams but struggled with more realistic and subtle ones.

Some research explored spam filters or phishing detection applied to job listings, but they typically relied on rule based or classical ML techniques and didn’t take advantage of deep learning models.

So, this project builds on those ideas by comparing three types of models:
- A naive baseline that always predicts “real”
- A classical model using TF-IDF with a Random Forest
- A deep learning model using DistilBERT fine-tuned on the job descriptions

The goal is to show how performance improves across these approaches and to make the tool easy for anyone to use through a web app.

## Modeling Approaches

I tested three different types of models

### 1. Naive Model
This model always predicts the majority class "real." It sets a simple baseline for performance.

### 2. Classical Model (Random Forest)
Uses TF-IDF to turn text into features, then trains a random forest. This model is fast and easy to interpret.

### 3. Deep Learning Model (DistilBERT)
Fine-tuned DistilBERT using the balanced dataset. It understands context and performs better on subtle or tricky scams.

## Training Details

Both the classical and deep models were trained on a balanced dataset. Since fake jobs are rare, I used upsampling to balance the classes. I trained the random forest on TF-IDF features and fine-tuned DistilBERT for one epoch using Hugging Face’s Trainer API.

## Evaluation

Models were evaluated using accuracy, precision, recall and F1 score. The results are shown below

### Model Comparison (Fraud Class)

| Model              | Precision | Recall | F1 Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| **Naive**          | 0.0000    | 0.0000 | 0.0000   | 0.9516   |
| **Classical**      | 0.9991    | 1.0000 | 0.9996   | 0.9996   |
| **Deep**           | 0.7085    | 0.9615 | 0.8159   | 0.9779   |

Note: Scores reflect performance on fake job posts only

- Naive: High accuracy but fails completely on fake jobs
- Classical: Nearly perfect on this test set but may be overfitting
- Deep: Great recall and solid F1, slightly lower precision but better at generalizing to trickier cases

In practice, the deep model offers the best balance for detecting scams while minimizing false alarms even if it’s not perfect. The classical model performs extremely well on this dataset but might struggle with new or subtle scams. The naive model exists mainly for comparison.

## Web App

The app is built with FastAPI and lets you paste a job description to get predictions from all three models. You can try it live here:
https://fake-job-detector-br91.onrender.com/

You can also use the API directly by sending a POST request to /predict. It accepts JSON with a job description and a model type (naive, classical, or deep). Only the deep model returns a confidence score.

Example request:
```
POST /predict
{
  "description": "Earn $1000/week from home, no experience needed",
  "model_type": "deep"
}
```

### Hugging Face

The deep learning model files are hosted on Hugging Face and downloaded at runtime by the app:
https://huggingface.co/dayeon3117/fake-job-detector-models/tree/main

The repo includes
- The final DistilBERT model in a .zip format
- config.json, vocab.txt, tokenizer files
- The original Kaggle dataset and prediction outputs for all models

## Running the App Locally

```bash
git clone https://github.com/dayeon3117/Fake-Job-Detector.git
cd Fake-Job-Detector
pip install -r requirements.txt

# Run the app locally
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then go to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

## Ethics and Limitations

This tool is designed to assist, not replace human judgment. While the detector can flag suspicious posts based on patterns in the training data, it may still miss cleverly written scams, especially ones that sound professional or avoid obvious red flags. It can also mislabel short, vague or unusual listings that don’t follow typical job post structures. The classical model may be overfitting to the training data and the deep model, while better at generalizing, is not perfect. Users should treat the results as a helpful signal.

The app doesn’t collect or store anything users submit. It runs entirely on public data and all predictions are transparent so users can interpret them however they see fit.

```
Fake-Job-Detector/
├── app/
│   ├── templates/
│   │   └── index.html           # HTML form for submitting job descriptions
│   ├── main.py                  # FastAPI UI route
├── data/
│   ├── raw/
│   │   └── fake_job_postings.csv      # Original Kaggle dataset (on Hugging Face)
│   └── outputs/                       # Prediction CSVs from all models (on Hugging Face)
│       ├── classical_predictions_rf.csv
│       ├── deep_predictions_balanced.csv
│       └── naive_predictions.csv
├── export_model/                # Final Hugging Face-ready DistilBERT model files
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── models/
│   ├── classical_model_rf.pkl
│   ├── naive_majority_class.pkl
│   └── vectorizer_rf.pkl
├── scripts/
│   ├── classical_model_rf.py
│   ├── compare_all_models.py
│   ├── evaluate_deep_model.py
│   ├── naive_model.py
│   └── train_deep_model.py
├── app.py                      # FastAPI JSON API 
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
```
