import torch
import torch.nn as nn
import torch.optim as optim

"""
In this file, we will implement a RNN cell from scratch.
"""

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        # Initialize the weights
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, h):
        # Update the hidden state
        h = torch.tanh(torch.mm(x, self.Wxh) + torch.mm(h, self.Whh) + self.bh)
        # Compute the output
        y = torch.mm(h, self.Why)
        return y, h
    
# Example parameters
batch_size = 1
input_size = 10
hidden_size = 20
output_size = 5

# Instantiate the model
model = CustomRNN(input_size, hidden_size, output_size)

# Initialize hidden state
hidden = torch.zeros(batch_size, hidden_size)

# Create a dummy input (e.g., one time step)
x = torch.randn(batch_size, input_size)

# Forward pass
y, new_hidden = model(x, hidden)

print(f"Output shape: {y.shape}")
print(f"New hidden state shape: {new_hidden.shape}")