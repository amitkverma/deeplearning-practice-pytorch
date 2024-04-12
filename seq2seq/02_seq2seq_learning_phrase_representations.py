import random
import torch
import torch.nn as nn
import torch.optim as optim
import helper
"""
In this file, we will implement a sequence-to-sequence model by following Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation paper.

Motivation:
In Previous implementation, we used the last hidden state of the encoder as the initial hidden state of the decoder.
One of the major drawbacks of this approach is that decoder might not be able to use the information from the encoder effectively when the input sequence is long.
In this implementation, we will use encoder's last hidden state with input and output sequences. So, the decoder can use the information from the encoder effectively.

The model architecture is as follows:
1. Encoder:
    - Embedding Layer
    - GRU Layer
2. Decoder:
    - Embedding Layer
    - GRU Layer
    - Linear Layer
    - Softmax Layer
3. Seq2Seq:
    - Encoder
    - Decoder
    - forward method
    
"""
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
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, embed dim]
        outputs, hidden = self.gru(embedded)
        # outputs = [src len, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]
        return hidden
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, dropout=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim)
        self.linear = nn.Linear(embed_dim + 2*hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target, hidden, context):
        # target = [batch size]
        # hidden = [1, batch size, hidden dim]
        # context = [1, batch size, hidden dim]
        target = target.unsqueeze(0)
        # target = [1, batch size]
        embedded = self.dropout(self.embedding(target))
        # embedded = [1, batch size, embed dim]
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, embed dim + hidden dim]
        output, hidden = self.gru(emb_con, hidden)
        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]
        prediction = self.linear(torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1))
        # prediction = [batch size, output dim]
        return prediction, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.linear.out_features
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        context = self.encoder(src)
        hidden = context
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs
    
# Define the model
embed_dim = 128
hidden_dim = 254
dropout = 0.5
device = helper.get_device()
lr = 1e-3

encoder = Encoder(len(en_vocab), embed_dim, hidden_dim, dropout).to(device)
decoder = Decoder(len(de_vocab), embed_dim, hidden_dim, dropout).to(device)
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

trainer.plot("Seq2Seq Learning Phrase Representations (Translation)")


# Console Output:
# 16.8% accuracy on test data
