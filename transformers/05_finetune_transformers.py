"""
In this file we will fine-tune a transformer model for a specific task using the Hugging Face Transformers library.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import evaluate

# Load the dataset
raw_dataset = load_dataset("imdb")
print(raw_dataset["train"].features) # check the features of the dataset
print(raw_dataset["train"][0]) # peek at the first training example

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(data):
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
# print(tokenized_datasets["train"][0]) # peek at the first training example after tokenization

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.with_format("torch")
# print(tokenized_datasets["train"][0]) # peek at the first training example after renaming columns

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define the compute_metrics function
def compute_metrics(eval_preds):
    metric = evaluate.load("imdb")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch"
)

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()



# Evaluate the model
results = trainer.evaluate()
print(results)
