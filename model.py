from transformers import pipeline, logging
from typing import List, Dict, Any

MODEL_PATH = "./models/finetuned_roberta"

def get_political_score(text):

    classifier = pipeline(
        "zero-shot-classification", 
        model=MODEL_PATH,
        return_all_scores=True,
    )
    
    # add a type check
    results: List[Dict[str, Any]] = classifier(text)[0] # type: ignore

    score_dict = {res['label']: res['score'] for res in results}

    weights = {"left": -1.0, "center": 0.0, "right": 1.0}
    weighted_score = sum(score_dict[label] * weights[label] for label in weights)

    return {
        "text": text,
        "score": round(weighted_score, 4),
        "probs": score_dict
    }

if __name__ == "__main__":
    sample_text = "The government should expand public healthcare access immediately."
    d = get_political_score(sample_text)

    # print("-1 (Extreme Left) to +1 (Extreme Right)")
    print(f"text: {d["text"]}")
    print(f"score: {d["score"]}")
    print(f"probs: {d["probs"]}")