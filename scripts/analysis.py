import os
import shap
import torch
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "matous-volf/political-leaning-politics"
tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict(texts):
    inputs = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

def save_result(shap_values, text, probs, title=""):

    os.makedirs("results", exist_ok=True)
    prefix = title if title else "analysis"
    
    # Indices for this specific model
    # 0: Left, 1: Center, 2: Right
    metadata = {
        "full_text": text,
        "base_values": {
            "left": float(shap_values.base_values[0][0]),
            "center": float(shap_values.base_values[0][1]),
            "right": float(shap_values.base_values[0][2])
        },
        "final_probabilities": {
            "left": float(probs[0][0]),
            "center": float(probs[0][1]),
            "right": float(probs[0][2])
        },
        "top_prediction": ["Left", "Center", "Right"][np.argmax(probs[0])]
    }
    
    with open(f"results/{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    token_data = {
        "token": shap_values.data[0],
        "shap_left": shap_values.values[0][:, 0],
        "shap_center": shap_values.values[0][:, 1],
        "shap_right": shap_values.values[0][:, 2]
    }
    df = pd.DataFrame(token_data)
    df.to_csv(f"results/{prefix}_tokens.csv", index=False)
    
    print(f"Full 3-class analysis saved for: {title}")

def analyze(text, title=""):
    explainer = shap.Explainer(predict, tokenizer)
    probs = predict(np.array([text]))
    shap_values = explainer([text])
    save_result(shap_values, text, probs, title)


if __name__ == "__main__":
    analyze("Tax cuts for corporations will result in increased economic activity and job opportunities.", "row_54")