import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import helper

"""
In this file, we will implement AlexNet architecture for CIFAR10 dataset. AlexNet was introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. It was the winning architecture of ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012.
In this implementation, we will follow some of the pytorch conventions and best practices:
1. We will use nn.Sequential to define the layers
2. We will use nn.ReLU(inplace=True) to save memory
3. We will use nn.Dropout() to prevent overfitting or regularization
4. We will use nn.init.kaiming_normal_ and nn.init.xavier_normal_ to initialize weights

AlexNet architecture is as follows:
1. Convolutional layer with 64 filters of size 3x3
2. Max pooling layer of size 2x2
3. Convolutional layer with 192 filters of size 3x3
4. Max pooling layer of size 2x2
5. Convolutional layer with 384 filters of size 3x3
6. Convolutional layer with 256 filters of size 3x3
7. Convolutional layer with 256 filters of size 3x3
8. Max pooling layer of size 2x2
9. Fully connected layer with 4096 units
10. Fully connected layer with 4096 units
11. Fully connected layer with 10 units (output layer)
"""

# Constants
batch_size = 64
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, input_shape = helper.load_dataset_image_data("CIFAR10" , batch_size)
train_loader, valid_loader ,test_loader = data

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
model = AlexNet().to(device)

helper.model_summary(model, input_shape)

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
        
model.apply(initialize_parameters)

# Hyperparameters
learning_rate = 0.001
epochs = 5

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
trainer = helper.Trainer(model, optimizer, criterion, lr=learning_rate, max_epochs=epochs)

trainer.train(train_loader, valid_loader)

trainer.plot("AlexNet")

trainer.test(test_loader)
