from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import os
import zipfile
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download, login

from scripts.classical_model_rf import get_rf_prediction
from scripts.naive_model import get_naive_prediction

# === Setup Hugging Face model path ===
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "dayeon3117/fake-job-detector-models"
ZIP_FILENAME = "distilbert_balanced.zip"
MODEL_DIR = "models/distilbert_balanced"

# === Download model from Hugging Face if not exists ===
if not os.path.exists(MODEL_DIR):
    print("Logging in to Hugging Face...") 
    login(token=HF_TOKEN)

    print("Downloading model zip from Hugging Face...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_FILENAME, repo_type="model")
    
    print("Extracting model zip...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("Model ready at:", MODEL_DIR)

# === Load model + tokenizer ===
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
deep_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
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
    # Deep model prediction
    deep_inputs = tokenizer(job_description, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        deep_outputs = deep_model(**deep_inputs)
        deep_pred = torch.argmax(deep_outputs.logits, dim=1).item()
    deep_result = "Suspicious" if deep_pred == 1 else "Real"

    # Other models
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
