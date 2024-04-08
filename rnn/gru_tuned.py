import torch
from torch import nn
import torch.optim as optim
import helper
import torchtext

"""
In this file, we will implement an advanced RNN model for sentiment analysis on the AG_NEWS dataset.

Motivation:
1. Use Pretrained Embeddings. (GloVe) - For better word representations.
2. Use Regualization techniques like Dropout. - To prevent overfitting.
3. Use Packed Sequence to handle variable length sequences. - For Variable length sequences.

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
glove = torchtext.vocab.GloVe(name='6B', dim=100)

use_pretrained_vectors = True
use_padded_sequence = True
data, num_class, vocab = helper.load_dataset_text_data('AG_NEWS', batch_size=64, tokenizer_type='basic_english', pretrained_vectors=glove)
train_loader, valid_loader, test_loader = data
vocab_size = len(vocab)


# Define the model
class RNNTunned(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,num_class, vocab_embeddings):
        super(RNNTunned, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(vocab_embeddings, freeze=True)
        self.gru = nn.GRU(input_size=embed_len, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=False, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.2)

    def forward(self, text):
        embeddings = self.dropout(self.embedding_layer(text))
        output, hidden = self.gru(embeddings)
        out = self.linear(output[:,-1])
        return out


# Define the hyperparameters
num_layers = 2
embed_len = 100
hidden_dim = 128
model = RNNTunned(vocab_size, embed_len, hidden_dim, num_layers,num_class, glove.vectors).to(device)
helper.model_summary(model)


    
# Define the loss function and optimizer
epochs = 20
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = helper.Trainer(model, optimizer, criterion, lr=learning_rate, max_epochs=epochs)

trainer.train(train_loader, valid_loader)

trainer.test(test_loader)

trainer.plot("GRU Tunned (AG_NEWS)")

# Console Output:
# 89.4% accuracy on the test set after 20 epochs Maybe because model have more parameters and it is underfitted