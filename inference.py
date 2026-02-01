import pandas as pd
from datasets import load_dataset
from model import analyze_political_leaning

# Load the dataset
dataset = load_dataset("cajcodes/political-bias")
df = pd.DataFrame(dataset['train']) # type: ignore

results = []
for index, row in df.iterrows():
    analysis = analyze_political_leaning(row['text'])
    
    results.append({
        "Text": row['text'],
        "True Label": row['label'],
        "Predicted Label": analysis['label'],
        "Model Score": round(analysis['score'], 4)
    })

# highly conservative (0) to highly liberal (4)
results_df = pd.DataFrame(results)
results_df.to_csv("results/sample_data.csv", index=False)