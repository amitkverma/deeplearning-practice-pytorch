"""
In this file we will fine-tune a transformer model for a specific task using the Hugging Face Transformers library.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import evaluate
import torch

# Function to predict sentiment
def predict_sentiment(model, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}  # Send input to the same device as model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return ["positive" if pred == 1 else "negative" for pred in predictions]

# Load the dataset
raw_dataset = load_dataset("imdb")
print(raw_dataset["train"].features) # Check the features of the dataset
print(raw_dataset["train"][0]) # Peek at the first training example

# Select a subset of the dataset
raw_dataset["train"] = raw_dataset["train"].select(range(1000))
raw_dataset["test"] = raw_dataset["test"].select(range(50))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(data):
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch")  # Set format for PyTorch

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Sample texts for testing model performance
sample_texts = ["This movie was absolutely wonderful!", "The worst movie I have ever seen."]

# Test model predictions before fine-tuning
print("Before fine-tuning:")
pre_tuning_predictions = predict_sentiment(model, sample_texts)
for text, sentiment in zip(sample_texts, pre_tuning_predictions):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

# Define the compute_metrics function
def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    logging_dir="./logs",
    evaluation_strategy="epoch"
)

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer)

eval_dataset = tokenized_datasets["test"]


# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Test model predictions after fine-tuning
print("After fine-tuning:")
post_tuning_predictions = predict_sentiment(model, sample_texts)
for text, sentiment in zip(sample_texts, post_tuning_predictions):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

# Save the fine-tuned model and tokenizer
model_name = "imdb_sentiment"
model.save_pretrained(f'{model_name}_model')
tokenizer.save_pretrained(f'{model_name}_tokken')
