import pandas as pd
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

def run_fine_tuning():
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
        eval_strategy="epoch",
        save_strategy="epoch",
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

        fp16=True,       
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

    trainer.train()
    trainer.save_model("./result/best_bias_model")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"mse": mse, "mae": mae}

if __name__ == "__main__":
    run_fine_tuning()