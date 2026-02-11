import torch
import torch.nn.functional as F
from transformers import pipeline, logging

logging.set_verbosity_error()

def get_political_score(text, model_path="roberta-large-mnli"):
    candidate_labels = ["Left-leaning", "Center", "Right-leaning"]
    weights = [-1.0, 0.0, 1.0]
    
    classifier = pipeline("zero-shot-classification", model=model_path)
    
    result = classifier(text, candidate_labels=candidate_labels)
    
    score_dict = dict(zip(result['labels'], result['scores']))
    ordered_probs = [score_dict[label] for label in candidate_labels]
    
    score = sum(p * w for p, w in zip(ordered_probs, weights))
    
    return {
        "text": text,
        "score": round(score, 4),
        "probs": score_dict
    }

if __name__ == "__main__":
    sample_text = "The government should expand public healthcare access immediately."
    d = get_political_score(sample_text)

    # print("-1 (Extreme Left) to +1 (Extreme Right)")
    print(f"text: {d["text"]}")
    print(f"score: {d["score"]}")
    print(f"probs: {d["probs"]}")