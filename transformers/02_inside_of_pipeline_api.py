"""
# In this file, we will understand inner workings of transformers pipeline API.

HuggingFace pipeline API can be broken down into the following components:
1. Preprocessing: Tokenization, padding, truncation, etc.
2. Model Inference: Get predictions from the model.
3. Postprocessing: Decoding, formatting, etc.

Let's understand these components in detail.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

texts = ["Hello, HuggingFace!", "Transformers are awesome!"]

# Step 1: Select a pre-trained model checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" 

# Step 2: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Step 3: Tokenize the input
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
print(encoded_inputs) 
# input_ids: Tokenized text
# token_type_ids: Segment IDs used in BERT
# attention_mask: Mask to avoid performing attention on padding tokens <pad> (0: ignore, 1: attend)


# Step 3: Load the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Step 4: Get the model's output
outputs = model(**encoded_inputs)

# Step 5. Softmax to get probabilities for each class
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# Step 6: Get the predicted label
predicted_class_idx = torch.argmax(predictions, dim=-1)
# Convert the predicted class index to label
predicted_label = predicted_class_idx.tolist()
out = [{"label": model.config.id2label[label], "score": score} for label, score in zip(predicted_label, predictions.tolist())]
print(out)