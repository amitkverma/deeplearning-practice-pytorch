import torch
from torch import nn
import torch.optim as optim
import helper

"""
In this file, we will implement an advanced RNN model for sentiment analysis on the AG_NEWS dataset.

Motivation:
1. Using a bidirectional RNN can capture the context from both directions of the input sequence.
2. Using multiple layers can help the model learn complex patterns in the data.

Architecture:
1. The model consists of an embedding layer, a bidirectional RNN layer, and a linear layer.
2. The embedding layer converts the input text into dense vectors of fixed size.
3. The bidirectional RNN layer processes the sequence data in both directions and passes the output to the linear layer.
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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,num_class):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(2* hidden_dim, num_class) # 2*hidden_dim because of bidirectional

    def forward(self, text):
        #text = [sent len, batch size]
        embeddings = self.embedding_layer(text)
        # embeddings = [sent len, batch size, embed dim]
        output, hidden = self.rnn(embeddings)
        # output shape: (batch_size, seq_len, hidden_dim) # contains hidden states for each time step
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim) # contains hidden states for the last time step or final hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # Concatenating the hidden states of the last layer
        # hidden shape: (batch_size, 2*hidden_dim)
        out = self.linear(hidden.squeeze(0))
        # out shape: (batch_size, num_class)
        return out


# Define the hyperparameters
num_layers = 2
embed_len = 80
hidden_dim = 128
model = RNNClassifier(vocab_size, embed_len, hidden_dim, num_layers,num_class).to(device)
helper.model_summary(model)


    
# Define the loss function and optimizer
epochs = 15
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = helper.Trainer(model, optimizer, criterion, lr=learning_rate, max_epochs=epochs)

trainer.train(train_loader, valid_loader)

trainer.test(test_loader)

trainer.plot("RNN Advanced (AG_NEWS)")

# Console Output:
# 90% accuracy on the test set after 15 epochs