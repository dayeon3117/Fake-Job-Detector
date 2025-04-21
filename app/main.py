from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from scripts.classical_model_rf import get_rf_prediction
from scripts.naive_model import get_naive_prediction

# === Initialize FastAPI and Jinja2 ===
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# === Load Deep Model ===
deep_model_path = "models/distilbert_balanced"
tokenizer = DistilBertTokenizer.from_pretrained("./models/distilbert_balanced", local_files_only=True)
deep_model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert_balanced", local_files_only=True)
deep_model.eval()

# === Route: Home Page ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === Route: HTML Form Prediction ===
@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, job_description: str = Form(...)):
    # Deep prediction
    deep_inputs = tokenizer(job_description, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        deep_outputs = deep_model(**deep_inputs)
        deep_pred = torch.argmax(deep_outputs.logits, dim=1).item()
    deep_result = "Suspicious" if deep_pred == 1 else "Real"

    # Random Forest + Naive
    classical_result = get_rf_prediction(job_description)
    naive_result = get_naive_prediction(job_description)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "deep_result": deep_result,
        "classical_result": classical_result,
        "naive_result": naive_result,
        "text": job_description
    })

# === Route: JSON API ===
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
