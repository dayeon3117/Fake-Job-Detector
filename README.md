# Fake-Job-Detector

Fake job listings are becoming more common and can lead to real problems like identity theft or money loss. Job platforms like LinkedIn and Indeed use their own tools to catch scams, but these tools are not public.

This project creates a public tool that lets anyone check if a job post seems suspicious. It uses different models to make predictions and runs through a simple web app that anyone can use.

## Problem Statement

Scam job listings are hard to spot, especially for people applying to jobs for the first time. This tool gives users a quick way to check a job description and see if it might be fake. It predicts if a listing is real or suspicious based on how it is written.

## Dataset

I used the [Fake Job Posting Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) from Kaggle. The dataset has job titles, descriptions and labels that show whether a job is real or fake.

The dataset is highly imbalanced; only about 5% of the job postings are labeled as fake. To deal with this, I balanced the training data using upsampling before training the classical and deep learning models.

## Background and Previous Approaches

Fake job posts have been a problem for a while. Many job platforms use their own tools or have people review listings by hand but those tools are not public.

Earlier efforts used basic machine learning models like Naive Bayes, support vector machines and logistic regression. These models looked for red flags like vague descriptions or missing contact details. They worked for some simple scams but struggled with more realistic ones.

In this project, I build on those ideas by comparing three types of models. I include a simple baseline that always predicts "real," a classical model using TF-IDF features with a random forest classifier and a deep learning model using DistilBERT that understands context better than keyword-based models.

The goal is to show how performance improves across these approaches and to make the tool easy for anyone to use through a web app.

## Modeling Approaches

I tested three different types of models

### 1. Naive Model
This model predicts that all job listings are real. It sets a simple baseline for performance.

### 2. Classical Model (Random Forest)
I use TF-IDF to turn the text into features and then train a random forest classifier. This model works well and is easy to understand.

### 3. Deep Learning Model (DistilBERT)
I fine-tuned a DistilBERT model using the balanced dataset. This model understands context and gives strong results, especially for recall.

## Training Details

The classical and deep learning models were both trained on a balanced dataset. Since fake job listings are much less common than real ones, I used upsampling to give both classes equal weight. For the classical model, I trained a random forest classifier on TF-IDF features. For the deep learning model, I fine-tuned DistilBERT for one epoch using Hugging Face’s Trainer API.

## Evaluation

I evaluated model performance using precision, recall, F1 score and accuracy on the fraudulent class. All metrics shown below refer to how well each model detected fake listings (Class 1).

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

This project is meant to help people avoid job scams, but it’s not perfect. It could mark some real jobs as fake or miss some actual scams. The results should help guide decisions but should not be taken as final answers.

I don't collect or store anything users type into the tool. The model was trained using public data and can be updated over time as scams change.


```
Fake-Job-Detector/
├── app/
│   ├── main.py                # FastAPI app UI route
│   └── templates/index.html
├── data/
│   ├── raw/                   # Raw Kaggle dataset
│   ├── outputs/               # Model predictions (CSV)
├── models/
│   ├── distilbert_balanced/  # Deep model checkpoints
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
├── .gitignore                # Ignore outputs, raw data, cache
├── requirements.txt          # Python dependencies
```