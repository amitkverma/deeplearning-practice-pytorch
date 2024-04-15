"""
In this file, we will explore the embeddings in transformers.
Embeddings are the vector representations of words or tokens in a model.
These embeddings are learned during the training process and capture the semantic meaning of words.
We can use these embeddings to compare the similarity between words or tokens.

In this example, we will use the BERT model to get the embeddings of different words and calculate the cosine similarity between them.
"""

from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Encode some text
tokens_king = tokenizer("king", return_tensors="pt")
tokens_queen = tokenizer("queen", return_tensors="pt")
tokens_transformer = tokenizer("transformer", return_tensors="pt")

# Get embeddings
with torch.no_grad():
    king_embedding = model(**tokens_king).last_hidden_state[:, 0, :]
    queen_embedding = model(**tokens_queen).last_hidden_state[:, 0, :]
    transformer_embedding = model(**tokens_transformer).last_hidden_state[:, 0, :]

# Calculate the cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(king_embedding, queen_embedding, dim=1)
print(f"Cosine similarity between 'king - queen': {cosine_similarity.item()}")

cosine_similarity = torch.nn.functional.cosine_similarity(king_embedding, transformer_embedding, dim=1)
print(f"Cosine similarity between 'king - transformer': {cosine_similarity.item()}")
