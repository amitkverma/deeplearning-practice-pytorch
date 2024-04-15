import random
import torch 
import torch.nn as nn
import torch.optim as optim
import helper
import time
# Set the seed
torch.manual_seed(0)

device = helper.get_device()

data, vocab = helper.load_translation_data(batch_size=128)
train_loader, valid_loader, test_loader = data
en_vocab, de_vocab = vocab


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,  num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, embed dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = [src len, batch size, hidden dim * num directions]
        # hidden = [num layers * num directions, batch size, hidden dim]
        # cell = [num layers * num directions, batch size, hidden dim]
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim ,num_layers, dropout=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        
    def forward(self, target, hideen, cell):
        # Since we are passing one sequence at a time, the sequence length is 1
        target = target.unsqueeze(0)
        # target = [1, batch size]
        embedded = self.dropout(self.embedding(target))
        # embedded = [1, batch size, embed dim]
        output, (hidden, cell) = self.lstm(embedded, (hideen, cell))
        # output = [1, batch size, hidden dim]
        prediction = self.linear(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = helper.get_device()
        
    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        input = trg[0,:] # pick the first token which is <sos>
        # input = [batch size]
        for t in range(1, trg_len): # skip the first token which is <sos>
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


# Define the hyperparameters
embed_dim = 64
hidden_dim = 128
layers = 2
dropout = 0.5
lr = 1e-3
teacher_forcing_ratio = 0.5

# Define the model
encoder = Encoder(len(en_vocab), embed_dim, hidden_dim, layers, dropout=dropout).to(device)
decoder = Decoder(len(de_vocab), embed_dim, hidden_dim, len(de_vocab), layers, dropout=dropout).to(device)

model = Seq2Seq(encoder, decoder, device).to(device)
helper.model_summary(model)

# Initialize the weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=de_vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=lr)

trainer = helper.Trainer(model, optimizer, criterion, lr=lr, max_epochs=10)

trainer.train(train_loader, valid_loader)

trainer.test(test_loader)

trainer.plot("Seq2Seq (Translation)")

# Console Output:
# 8% accuracy on the test set