import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from utils import format_cajcodes_political_bias

output_path = "./models/finetuned_roberta"
id2label = {0: "left", 1: "center", 2: "right"}
label2id = {"left": 0, "center": 1, "right": 2}

# using cajcodes_political_bias as testing for finetune
# have to download raw cajcodes_political_bias to there first
raw_data_path = './data/sample_data'
formatted_data = './data/formatted_sample_data'

data_path = format_cajcodes_political_bias(raw_data_path, formatted_data)

# need pure csv file as finetuning data
# 1st column: text 
# 2nd column: label 
df = pd.read_csv(data_path)
df['label'] = df['label'].map(label2id)
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.1, seed=42)

# load roberta model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# preprocess data
def tokenize_func(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_func, batched=True)

# load model for 3-classes classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# define training
training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

# save the model for inference
trainer.save_model(output_path)