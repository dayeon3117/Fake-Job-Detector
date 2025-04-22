from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import login

from scripts.classical_model_rf import get_rf_prediction
from scripts.naive_model import get_naive_prediction

# === Hugging Face setup ===
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "dayeon3117/fake-job-detector-models"

# === Login to Hugging Face ===
print("Logging in to Hugging Face...")
login(token=HF_TOKEN)

# === Load model and tokenizer from HF Hub root ===
print("Loading model and tokenizer from Hugging Face Hub...")
tokenizer = DistilBertTokenizer.from_pretrained(REPO_ID)
deep_model = DistilBertForSequenceClassification.from_pretrained(REPO_ID)
deep_model.eval()

# === Initialize FastAPI app ===
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# === Route: Home Page ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === Route: HTML Form Prediction ===
@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, job_description: str = Form(...)):
    deep_inputs = tokenizer(job_description, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        deep_outputs = deep_model(**deep_inputs)
        deep_pred = torch.argmax(deep_outputs.logits, dim=1).item()
    deep_result = "Suspicious" if deep_pred == 1 else "Real"

    classical_result = get_rf_prediction(job_description)
    naive_result = get_naive_prediction(job_description)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "deep_result": deep_result,
        "classical_result": classical_result,
        "naive_result": naive_result,
        "text": job_description
    })

# === Route: JSON API Prediction ===
@app.post("/api/predict")
async def api_predict(job_description: str = Form(...)):
    deep_inputs = tokenizer(job_description, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        deep_outputs = deep_model(**deep_inputs)
        deep_pred = torch.argmax(deep_outputs.logits, dim=1).item()
    deep_result = "Suspicious" if deep_pred == 1 else "Real"

    classical_result = get_rf_prediction(job_description)
    naive_result = get_naive_prediction(job_description)

    return JSONResponse({
        "job_description": job_description,
        "deep_model": deep_result,
        "classical_model": classical_result,
        "naive_model": naive_result
    })
