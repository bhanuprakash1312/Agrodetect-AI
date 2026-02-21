import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import google.generativeai as genai
from huggingface_hub import hf_hub_download
import re

REPO_ID = "bhanu-13/Agrodetect-AI"
genai.configure(api_key="AIzaSyBWL00-84wRaC1QyaVJs42ee3jCY7KBfIE")
model_gemini = genai.GenerativeModel("gemini-flash-lite-latest")

def load_model():
    # Download files from Hugging Face
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", repo_type="space")
    class_path = hf_hub_download(repo_id=REPO_ID, filename="class_names.json", repo_type="space")

    # Load class labels
    with open(class_path) as f:
        class_names = json.load(f)

    # Transforms
    global leaf_transform
    leaf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])

    # Load model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, class_names

def predict_disease(model, class_names, image: Image.Image):
    print("[DEBUG] Image received for prediction.")
    image = image.convert("RGB")
    input_tensor = leaf_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        print("[DEBUG] Model output:", output)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

def load_leaf_classifier():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="leaf_binary_classifier.pth", repo_type="space")

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model

def get_treatment(disease):
    import os
    treatment_path = os.path.join(os.path.dirname(__file__), "../disease_treatments.json")
    with open(treatment_path) as f:
        treatment_dict = json.load(f)
    return treatment_dict.get(disease, "No treatment recommendation available.")
def is_leaf_image(model, image: Image.Image, threshold: float = 0.6) -> bool:

    # FIX 1: Always convert to RGB
    image = image.convert("RGB")

    # FIX 2: Apply transform correctly
    input_tensor = leaf_transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        # DEBUG: see probabilities
        print("[DEBUG] Probabilities:", probabilities)

        # FIX 3: leaf class index = 0 (assuming training order)
        leaf_confidence = probabilities[0][0].item()

        print(f"[DEBUG] Leaf confidence: {leaf_confidence:.4f}")

        # FIX 4: lower threshold slightly for better detection
        return leaf_confidence >= threshold
def translate_disease_info(prediction: str, treatment: str, language: str):

    # Skip if English
    if language.lower() == "english":
        return prediction, treatment

    prompt = f"""
Translate into {language}.

Return EXACTLY this format:

Disease: <translated disease>
Treatment: <translated treatment>

Disease: {prediction}
Treatment: {treatment}
"""

    try:

        response = model_gemini.generate_content(prompt)

        text = response.text.strip()

        print("[DEBUG Gemini translation]:", text)

        # Remove markdown if present
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Split safely
        disease_part = prediction
        treatment_part = treatment

        if "Disease:" in text:
            disease_part = text.split("Disease:")[1].split("Treatment:")[0].strip()

        if "Treatment:" in text:
            treatment_part = text.split("Treatment:")[1].strip()

        return disease_part, treatment_part

    except Exception as e:

        print("[ERROR Gemini translation]:", e)

        return prediction, treatment
# utils.py


def generate_actions_and_prevention(disease: str, language: str):

    prompt = f"""
    Provide crop disease management for: {disease}

    Return ONLY valid JSON. No extra text.

    Format:
    {{
      "actions": [
        "action 1",
        "action 2",
        "action 3"
      ],
      "prevention": [
        "prevention 1",
        "prevention 2",
        "prevention 3"
      ]
    }}

    Rules:
    - Language: {language}
    - Simple farmer-friendly language
    - Max 8 words per point
    - Exactly 3 actions and 3 prevention
    """

    try:

        response = model_gemini.generate_content(prompt)

        text = response.text.strip()

        # remove markdown if exists
        text = text.replace("```json", "").replace("```", "")

        data = json.loads(text)

        return data["actions"], data["prevention"]

    except Exception as e:

        print("[ERROR] Gemini actions generation failed:", e)

        return [], []