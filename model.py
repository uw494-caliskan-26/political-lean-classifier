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
    p_right = probs[2].item()
    
    # Calculate Continuous Score
    continuous_score = (p_left - p_right + 1) / 2

    # Identify the specific predicted label
    labels = ["Left", "Center", "Right"]
    predicted_index = torch.argmax(probs).item()
    predicted_label = labels[predicted_index] # type: ignore
    
    return {
        "text": text,
        "score": continuous_score,
        "label": predicted_label,
    }


if __name__ == "__main__":
    # --- Example Usage ---
    sample_text = "The government should reduce taxes significantly because it will stimulate economic growth and allow businesses to thrive."
    result = analyze_political_leaning(sample_text)

    # --- Output ---
    print(f"Text: \"{result['text']}\"\n")
    print(f"Score: {result['score']:.4f} (1 Right, 0 Left)")
    print(f"Predicted Label:  {result['label']}")