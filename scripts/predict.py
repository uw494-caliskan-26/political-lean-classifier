import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Update this path to the fine-tuned model directory
MODEL_DIR = "./models/finetuned_roberta"

def load_model(model_dir=MODEL_DIR, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device


def predict_score(text, tokenizer=None, model=None, device=None):
    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    score = outputs.logits.squeeze().item()
    return score


if __name__ == "__main__":
    sample = "Tax cuts for corporations will result in increased economic activity."
    print(predict_score(sample))
