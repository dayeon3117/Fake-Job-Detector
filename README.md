# Fake-Job-Detector

Fake job listings are becoming more common and can lead to serious problems like identity theft or money loss. Job platforms like LinkedIn and Indeed use their own tools to catch scams, but these tools are not public.

This project creates a public tool that lets anyone check if a job post seems suspicious. It uses different models to make predictions and runs through a simple web app that anyone can use.

## Problem Statement

Scam job listings are hard to spot, especially for people applying to jobs for the first time. This tool gives users a quick way to check a job description to see if it might be fake. It predicts if a listing is real or suspicious based on how it is written.

## Dataset

I used the [Fake Job Posting Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) from Kaggle. The dataset has job titles, descriptions and labels showing whether a job is real or fake.

The dataset is highly imbalanced; only about 5% of the job postings are labeled as fake. To deal with this, I balanced the training data using upsampling before training the classical and deep learning models.

## Background and Previous Approaches

Fake job posts have been around for a long time. Many platforms rely on in-house tools or manual reviews, but these are not available to the public.

Earlier efforts used basic machine learning models like Naive Bayes, support vector machines and logistic regression. These models looked for red flags like vague descriptions or missing contact details. They worked for some simple scams but struggled with more realistic ones.

This project builds on those ideas by comparing three types of models:
- A naive baseline that always predicts “real”
- A classical model using TF-IDF and a random forest classifier
- A deep learning model using DistilBERT for better context understanding

The goal is to show how performance improves across these approaches and to make the tool easy for anyone to use through a web app.

## Modeling Approaches

I tested three different types of models

### 1. Naive Model
This model predicts that all job listings are real. It sets a simple baseline for performance.

### 2. Classical Model (Random Forest)
Uses TF-IDF to turn text into features, then trains a random forest. This model is fast and easy to interpret.

### 3. Deep Learning Model (DistilBERT)
Fine-tuned DistilBERT using the balanced dataset. It understands context and performs well, especially on recall.

## Training Details

Both the classical and deep models were trained on a balanced dataset. Since fake jobs are rare, I used upsampling to balance the classes. I trained the random forest on TF-IDF features and fine-tuned DistilBERT for one epoch using Hugging Face’s Trainer API.

## Evaluation

Models were evaluated using accuracy, precision, recall and F1 score. The results below focus on the model’s ability to detect fake jobs (Class 1).

### Model Comparison (Fraud Class)

| Model              | Precision | Recall | F1 Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| **Naive**          | 0.0000    | 0.0000 | 0.0000   | 0.9516   |
| **Classical**      | 0.9991    | 1.0000 | 0.9996   | 0.9996   |
| **Deep**           | 0.7085    | 0.9615 | 0.8159   | 0.9779   |

Note: Scores reflect performance on fake (fraudulent) job posts only

## Web App

I built the app using FastAPI. It has two ways to use it:

- Go to the home page and paste in a job description to get results
- Use the API to send in descriptions and get predictions in return

Each prediction shows  
- Whether the job looks real or suspicious from all three models 
- A confidence score for the deep learning model

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

## Ethics Statement

This project is meant to help people avoid scams, but it’s not perfect. It may incorrectly mark a real job as fake or miss an actual scam. The results are meant to guide users, not make final decisions.

The tool does not collect or store anything users enter. All models were trained on public data and can be retrained in the future as scam patterns evolve.

```
Fake-Job-Detector/
├── app/
│   ├── main.py                # FastAPI app UI route
│   └── templates/index.html
├── data/
│   ├── raw/                   # Original dataset
│   ├── outputs/               # CSV model predictions
├── models/
│   ├── distilbert_balanced/  # Fine-tuned deep model
│   ├── classical_model_rf.pkl
│   └── naive_majority_class.pkl
├── scripts/
│   ├── train_deep_model.py
│   ├── evaluate_deep_model.py
│   ├── classical_model_rf.py
│   ├── naive_model.py
│   └── compare_all_models.py
├── app.py                    # JSON API route
├── README.md                 # Project documentation
├── .gitignore                # Ignore files
├── requirements.txt          # Python dependencies
```
