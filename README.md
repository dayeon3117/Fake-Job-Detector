# Fake-Job-Detector

Fake job postings are a growing issue on job boards and career sites. These scams often look real and can lead to serious problems like identity theft or financial loss. Right now, most job platforms rely on manual reviews or internal AI systems, but those tools are not available to the public and usually do not explain why something was flagged.

This project builds an AI-powered tool to help users check if a job post might be fake. It classifies whether a listing is real or suspicious and explains why using language models and basic explainability techniques.

## Problem Statement

Fake job listings are hard to spot and can cause real harm. Many people apply to these scams without realizing they are not legitimate. Most job sites do not offer an easy way to check if a post is trustworthy. This project provides a public tool that uses machine learning to flag risky listings and explain what made them suspicious.

## Dataset

We use the [Fake Job Posting Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) from Kaggle. The dataset includes job descriptions, company names, locations and labels that show if a post is real or fake. It works well for building a binary classifier.

We also include patterns found in real scam examples like listings that ask for payment or leave out important contact details. These patterns help improve how the model explains its results.

## Modeling Approaches

This project compares three types of models

### 1. Naive Model
This is a simple baseline that always predicts the most common label, usually "real." It helps show how much better the other models perform.

### 2. Classical Machine Learning Model
We use logistic regression with TF-IDF features from the job descriptions. This model is fast to train and easy to understand. It works well as a middle ground between the baseline and deep learning models.

### 3. Deep Learning Model
We use a fine-tuned BERT model from Hugging Face. It looks at the full context of the job description to make predictions. This model usually gives the best results on text tasks like this one.

### Model Explainability
We use SHAP to show which words or phrases had the biggest impact on the model’s decision. This makes it easier to understand why a listing was flagged as fake.

## Evaluation

We evaluate the models based on:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Each result is shown in the code and summary files. This helps explain the tradeoffs between the different models.

## Web App

The tool is built with FastAPI and deployed to the cloud. Users can paste a job description into the app and see

- A prediction: Real or Suspicious  
- A confidence score  
- A short explanation showing what raised concern  

The app is simple to use and does not require any technical background.

## How to Run the Project

To run the project locally:

```bash
git clone https://github.com/dayeon3117/Fake-Job-Detector.git
cd Fake-Job-Detector
pip install -r requirements.txt
uvicorn app:app --reload
```

Once the app is running, open your browser and go to:

http://127.0.0.1:8000

## Ethics Statement

This tool is designed to help people avoid job scams, but it is not perfect. Some real listings might be flagged by mistake and some fake ones might not be caught. Users should treat the results as advice, not a final answer.

The tool does not collect or save anything users enter. We also understand that scammers change their tactics over time, so we plan to retrain the model when new patterns appear.


