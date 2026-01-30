import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load Model and Tokenizer
model_name = "matous-volf/political-leaning-politics"
tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_political_leaning(text):
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert logits to probabilities (0: Left, 1: Center, 2: Right)
        probs = F.softmax(logits, dim=-1).squeeze()

    # Extract specific probabilities
    p_left = probs[0].item()
    p_center = probs[1].item()
    p_right = probs[2].item()
    
    # Calculate Continuous Score
    continuous_score = p_right - p_left

    # Identify the specific predicted label
    labels = ["Left", "Center", "Right"]
    predicted_index = torch.argmax(probs).item()
    predicted_label = labels[predicted_index] # type: ignore
    predicted_probability = probs[predicted_index].item() # type: ignore
    
    return {
        "text": text,
        "score": continuous_score,
        "predicted_label": predicted_label,
        "predicted_probability": predicted_probability,
        "probabilities": {
            "Left": f"{p_left:.4f}",
            "Center": f"{p_center:.4f}",
            "Right": f"{p_right:.4f}"
        }
    }


if __name__ == "__main__":
    # --- Example Usage ---
    sample_text = "The government should reduce taxes significantly because it will stimulate economic growth and allow businesses to thrive."
    result = analyze_political_leaning(sample_text)

    # --- Output ---
    print(f"Text: \"{result['text']}\"\n")
    print(f"Continuous Score: {result['score']:.4f} (Scale: -1 Left to +1 Right)")
    print(f"Predicted Label:  {result['predicted_label']} ({result['predicted_probability']:.2%} confidence)")
    print(f"Raw Probabilities: {result['probabilities']}")