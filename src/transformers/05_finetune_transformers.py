"""
In this file we will train a transformer model for a sentiment analysis using the Hugging Face Transformers library.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import evaluate
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
tokenized_datasets = tokenized_datasets.with_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_datasets["train"], batch_size=64, collate_fn=data_collator, shuffle=True)
eval_loader = DataLoader(tokenized_datasets["test"], batch_size=64, collate_fn=data_collator)

# Define the model and load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

# Define hyperparameters
num_epochs = 3

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}: Average Training Loss: {total_loss / len(train_loader)}")

    # Evaluate the model
    model.eval()
    metric = evaluate.load("accuracy")
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    print(f"Validation Accuracy: {metric.compute()}")

# Save the model
model_name = "imdb_sentiment_model"
model.save_pretrained(model_name)


# Load a saved model
trained_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Function to predict sentiment
def predict_sentiment(texts):
    # Prepare the texts for the model
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

    # Predict
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(**tokenized_inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    return ["positive" if pred == 1 else "negative" for pred in predictions]

# Sample texts for prediction
sample_texts = ["This movie was absolutely wonderful!", "The worst movie I have ever seen."]
predicted_sentiments = predict_sentiment(sample_texts)
for text, sentiment in zip(sample_texts, predicted_sentiments):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
