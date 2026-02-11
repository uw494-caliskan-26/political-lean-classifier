from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# 1. Prepare your data (Assuming a CSV with 'text' and 'label' columns)
# Labels: 0=Left, 1=Center, 2=Right
df = pd.read_csv("political_data.csv")
dataset = Dataset.from_pandas(df)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_func(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_func, batched=True)

# 2. Load Model for 3-class Classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./political_roberta",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# 4. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 5. Start Training
trainer.train()

# 6. Save the model for inference
trainer.save_model("./finetuned_political_model")