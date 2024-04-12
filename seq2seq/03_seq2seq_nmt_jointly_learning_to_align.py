""" 
In this file, we will implement Machine Translation by Jointly Learning to Align and Translate paper. Which is an extension of seq2seq model.

Motivation:
In previous implementation, we used the last hidden state of the encoder to concatenate with the decoder's hidden state to predict the next word. 
Instead of using the last hidden state of the encoder, we will use the attention mechanism to align the input and output sequences.

"""


import random
import torch
import torch.nn as nn
import torch.optim as optim
import helper
import torch.nn.functional as F

# Set the seed
torch.manual_seed(0)

device = helper.get_device()

data, vocab = helper.load_translation_data(batch_size=128)
train_loader, valid_loader, test_loader = data
en_vocab, de_vocab = vocab


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, embed dim]
        outputs, hidden = self.gru(embedded)
        # outputs = [src len, batch size, hidden dim]
        # hidden = [2, batch size, hidden dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, hidden dim]
        # encoder_outputs = [src len, batch size, hidden dim]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch size, src len, hidden dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, 2*hidden dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hidden dim]
        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        return F.softmax(attention, dim=0)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention ,dropout=0.2):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.gru = nn.GRU(embed_dim + 2*hidden_dim, hidden_dim)
        self.fc = nn.Linear(embed_dim + 2*hidden_dim + hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, hidden dim]
        # encoder_outputs = [src len, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embed dim]
        attention_weights = self.attention(hidden, encoder_outputs)
        # attention_weights = [batch size, src len]
        attention_weights = attention_weights.unsqueeze(1)
        # attention_weights = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, 2*hidden dim]
        weighted = torch.bmm(attention_weights, encoder_outputs)
        # weighted = [batch size, 1, 2*hidden dim]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, 2*hidden dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, embed dim + 2*hidden dim]
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))
        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]
        prediction = self.fc(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0)
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs
    
input_dim = len(en_vocab)
output_dim = len(de_vocab)
embed_dim = 128
hidden_dim = 256
dropout = 0.5
lr = 0.001

encoder = Encoder(input_dim, embed_dim, hidden_dim, dropout).to(device)
attention = Attention(hidden_dim).to(device)
decoder = Decoder(output_dim, embed_dim, hidden_dim, attention, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

helper.model_summary(model)

# Initialize the weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

# Define the optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=de_vocab["<pad>"])

# Train the model
max_epochs = 10
clip = 1
teacher_forcing_ratio = 0.5
trainer = helper.Trainer(model, optimizer, criterion, max_epochs=max_epochs)

trainer = helper.Trainer(model, optimizer, criterion, lr=lr, max_epochs=max_epochs)

trainer.train(train_loader, valid_loader)

trainer.test(test_loader)

# Save the model
trainer.plot("Seq2Seq NMT (Jointly Learning to Align and Translate)")