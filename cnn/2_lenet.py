import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import helper
# from helper import load_dataset_image_data, Trainer

"""
In this file, we will implement simple LeNet architecture for MNIST dataset. LeNet was very first CNN architecture introduced by Yann LeCun for handwritten digit recognition.

LeNet architecture is as follows:
1. Convolutional layer with 6 filters of size 5x5
2. Max pooling layer of size 2x2
3. Convolutional layer with 16 filters of size 5x5
4. Max pooling layer of size 2x2
5. Fully connected layer with 120 units
6. Fully connected layer with 84 units
7. Fully connected layer with 10 units (output layer)

As, we noticed LeNet architecture follows Conv -> Pool Layers with increasing number of filters and decreasing size of filters. 
Finally, we have fully connected layers to classify the output.
"""

# Constants
batch_size = 124
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, input_shape = helper.load_dataset_image_data("MNIST" , batch_size)
train_loader, valid_loader ,test_loader = data

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # x shape: 64 x 1 x 28 x 28
        out = self.conv1(x)
        # out share: 64 x 6 x 24 x 24
        out = F.max_pool2d(out, 2)
        # out shape: 64 x 6 x 12 x 12
        out = F.relu(out)
        # out shape: 64 x 6 x 12 x 12
        out = self.conv2(out)
        # out shape: 64 x 16 x 8 x 8
        out = F.max_pool2d(out, 2)
        # out shape: 64 x 16 x 4 x 4
        out = F.relu(out)
        # out shape: 64 x 16 x 4 x 4
        out = out.view(out.size(0), -1)
        # out shape: 64 x 256
        out = self.fc1(out)
        # out shape: 64 x 120
        out = F.relu(out)
        # out shape: 64 x 120
        out = self.fc2(out)
        # out shape: 64 x 84
        out = F.relu(out)
        # out shape: 64 x 84
        out = self.fc3(out)
        # out shape: 64 x 10
        return out


model = LeNet().to(device)
helper.model_summary(model, input_shape)

# Hyperparameters
learning_rate = 0.001
epochs = 5

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
trainer = helper.Trainer(model, optimizer, criterion, lr=learning_rate, max_epochs=epochs)

trainer.train(train_loader, valid_loader)

trainer.plot("LeNet")

trainer.test(test_loader)
