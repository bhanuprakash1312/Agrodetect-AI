from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from backend.utils import (
    load_model,
    load_leaf_classifier,
    is_leaf_image,
    predict_disease,
    get_treatment
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://crop-disease-detector-olive.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
disease_model, class_names = load_model()
leaf_model = load_leaf_classifier()

@app.post("/verify-leaf")
async def verify_leaf(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    if not is_leaf_image(leaf_model, image):
        raise HTTPException(status_code=400, detail="Uploaded image is not a leaf.")

    return {"isLeaf": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    if not is_leaf_image(leaf_model, image):
        raise HTTPException(status_code=400, detail="Uploaded image is not a leaf.")

    disease = predict_disease(disease_model, class_names, image)
    treatment = get_treatment(disease)

    return {
        "prediction": disease,
        "treatment": treatment
    }


@app.get("/ping")
def ping():
    return {"status": "alive"}
