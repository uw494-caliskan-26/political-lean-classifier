import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Plot label and model prediction distributions.")
    parser.add_argument("--data-path", default="./data/clean_data.parquet", help="Path to parquet dataset.")
    parser.add_argument("--model-dir", default="./results/best_bias_model", help="Path to fine-tuned model.")
    parser.add_argument("--output-dir", default="./results/distribution_plots", help="Directory for output plots.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI.")
    return parser.parse_args()


def load_model(model_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def batched_predict(texts, tokenizer, model, device, batch_size):
    all_predictions = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_predictions = outputs.logits.squeeze(-1).detach().cpu().numpy()
        all_predictions.extend(batch_predictions.tolist())

    return pd.Series(all_predictions, name="prediction")


def save_distribution_plot(values, title, x_label, output_file, bins, dpi):
    plt.figure(figsize=(10, 5))
    sns.histplot(values, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()


def save_overlay_plot(labels, predictions, output_file, bins, dpi):
    plt.figure(figsize=(10, 5))
    sns.histplot(labels, bins=bins, stat="density", color="steelblue", alpha=0.4, label="Labels")
    sns.histplot(predictions, bins=bins, stat="density", color="tomato", alpha=0.4, label="Predictions")
    plt.title("Label vs Prediction Distribution")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()


def save_summary(labels, predictions, output_file):
    summary = pd.DataFrame(
        {
            "stat": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            "label": labels.describe().reindex(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]).values,
            "prediction": predictions.describe().reindex(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]).values,
        }
    )
    summary.to_csv(output_file, index=False)


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    df = pd.read_parquet(data_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset missing columns: {', '.join(sorted(missing_columns))}")

    labels = pd.Series(pd.to_numeric(df["label"], errors="coerce"), name="label").dropna()
    inference_df = df.loc[labels.index]

    tokenizer, model, device = load_model(str(model_dir))
    predictions = batched_predict(
        texts=inference_df["text"].astype(str).tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    label_plot = output_dir / "label_distribution.png"
    prediction_plot = output_dir / "prediction_distribution.png"
    overlay_plot = output_dir / "label_prediction_overlay.png"
    summary_csv = output_dir / "distribution_summary.csv"

    save_distribution_plot(labels, "Label Distribution", "Label", label_plot, args.bins, args.dpi)
    save_distribution_plot(predictions, "Prediction Distribution", "Prediction", prediction_plot, args.bins, args.dpi)
    save_overlay_plot(labels, predictions, overlay_plot, args.bins, args.dpi)
    save_summary(labels, predictions, summary_csv)

    print(f"Rows used: {len(inference_df)}")
    print(f"Device: {device}")
    print(f"Saved: {label_plot}")
    print(f"Saved: {prediction_plot}")
    print(f"Saved: {overlay_plot}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
