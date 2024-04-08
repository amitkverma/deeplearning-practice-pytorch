import torch
from torch import nn
import torch.optim as optim
import helper

"""
In this file, we will implement a simple RNN model for sentiment analysis on the AG_NEWS dataset.

Motivation:
1. RNNs are a class of neural networks that allow previous outputs to be used as inputs while having hidden states.
2. They are suitable for sequence data and are used in various applications like time series analysis, speech recognition, etc.

Architecture:
1. The model consists of an embedding layer, an RNN layer, and a linear layer.
2. The embedding layer converts the input text into dense vectors of fixed size.
3. The RNN layer processes the sequence data and passes the output to the linear layer.
4. The linear layer produces the output logits.


"""

# Set the seed
torch.manual_seed(0)
device = helper.get_device()

# Load the dataset
data, num_class, vocab = helper.load_dataset_text_data('AG_NEWS', batch_size=64, tokenizer_type='basic_english')
train_loader, valid_loader, test_loader = data
vocab_size = len(vocab)

# Define the model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_class)

    def forward(self, text):
        embeddings = self.embedding_layer(text)
        output, hidden = self.rnn(embeddings)
        # output shape: (batch_size, seq_len, hidden_dim) # contains hidden states for each time step
        # hidden shape: (num_layers, batch_size, hidden_dim) # contains hidden states for the last time step or final hidden states
        out = self.linear(output[:,-1])
        return out


# Define the hyperparameters
embed_len = 80
hidden_dim = 128
model = RNNClassifier(vocab_size, embed_len, hidden_dim, num_class).to(device)
helper.model_summary(model)


    
# Define the loss function and optimizer
epochs = 15
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = helper.Trainer(model, optimizer, criterion, lr=learning_rate, max_epochs=epochs)

trainer.train(train_loader, valid_loader)

trainer.test(test_loader)

trainer.plot("RNN Sentiment Analysis (AG_NEWS)")

# Console Output:
# 86% accuracy on the test set after 15 epochs