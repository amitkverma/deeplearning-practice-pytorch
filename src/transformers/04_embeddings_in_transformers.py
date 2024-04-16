"""
In this file, we will explore the embeddings in transformers.
Embeddings are the vector representations of words or tokens in a model.
These embeddings are learned during the training process and capture the semantic meaning of words.
We can use these embeddings to compare the similarity between words or tokens.

In this example, we will use the BERT model to get the embeddings of different words and calculate the cosine similarity between them.
"""

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("imdb", split="train")
dataset = dataset.select(range(1))
# Load the BERT model and tokenizer
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1" # Good for sentence embeddings
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

# Using CLS pooling to get the sentence embeddings Where we collect the last hidden state for the special [CLS] token.
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

text = ["I am walking in the park."]
embedding = get_embeddings(text)
print(embedding.shape)

embedding_dataset = dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

# Using FAISS for efficient similarity search

embedding_dataset.add_faiss_index(column="embeddings")


question = "How was godfather movie?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape

scores, samples = embedding_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
