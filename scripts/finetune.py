from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_NAME = "roberta-base"
DATASET_PATH = "./data/clean_data.parquet"
MODEL_OUTPUT_DIR = "./results/best_bias_model"
METRICS_OUTPUT_PATH = "./results/logs/training_metrics.csv"

def save_training_metrics(log_history, output_path=METRICS_OUTPUT_PATH):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Keep all trainer logs while ensuring core metric columns are easy to query.
    metrics_df = pd.DataFrame(log_history)
    core_columns = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    for column in core_columns:
        if column not in metrics_df.columns:
            metrics_df[column] = pd.NA

    ordered_cols = core_columns + [col for col in metrics_df.columns if col not in core_columns]
    metrics_df = metrics_df[ordered_cols]
    metrics_df.to_csv(output_file, index=False)
    return str(output_file)

def run_fine_tuning():
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    # prepare dataset
    df = pd.read_parquet(DATASET_PATH)
    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding=True, max_length=512), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    training_args = TrainingArguments(
        output_dir="./roberta-bias-regression",
        eval_strategy="steps",   # Evaluate based on steps, not epochs
        eval_steps=100,          # Run evaluation every 100 steps
        save_strategy="steps",   # Save strategy must match eval strategy if load_best_model_at_end=True
        save_steps=1000,         # Save checkpoint every 1000 steps
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        save_total_limit=2, 
        logging_steps=10,

        fp16=use_cuda,
        dataloader_num_workers=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Trainer resolved device: {trainer.args.device}")
    print(f"Trainer n_gpu: {trainer.args.n_gpu}")

    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    metrics_path = save_training_metrics(trainer.state.log_history)
    print(f"Saved training metrics to: {metrics_path}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"mse": mse, "mae": mae}

if __name__ == "__main__":
    run_fine_tuning()