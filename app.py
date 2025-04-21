from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import joblib

# Import classical + naive prediction functions
from scripts.classical_model_rf import get_rf_prediction as get_classical_prediction
from scripts.naive_model import get_naive_prediction

# Define input format
class JobInput(BaseModel):
    description: str
    model_type: str  # "naive", "classical", "deep"

# Load deep learning model
tokenizer = DistilBertTokenizer.from_pretrained("export_model")
deep_model = DistilBertForSequenceClassification.from_pretrained("export_model")
deep_model.eval()

# Init app
app = FastAPI(title="Fake Job Detector")

@app.post("/predict")
def predict(input: JobInput):
    try:
        text = input.description
        model_type = input.model_type.lower()

        if model_type == "naive":
            label = get_naive_prediction(text)
            return {"label": label, "model": "Naive"}
        
        elif model_type == "classical":
            label = get_classical_prediction(text)
            return {"label": label, "model": "Classical"}
        
        elif model_type == "deep":
            # Tokenize and predict
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = deep_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()
            label = "Suspicious" if pred_class == 1 else "Real"
            return {"label": label, "confidence": round(confidence, 4), "model": "Deep Learning (DistilBERT)"}
        
        else:
            raise ValueError("Invalid model type: must be 'naive', 'classical', or 'deep'")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
