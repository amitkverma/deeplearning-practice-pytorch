import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import helper

"""
In this file, we will implement GoogleNet architecture for CIFAR10 dataset. GoogleNet was introduced by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna in 2014. 
It was the winning architecture of ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014.

Motivation::
1. Increase Depth and Width Without Overfitting: it aimed to deepen and widen the network while including regularization techniques to control overfitting.
2. Reduce Parameter Count : This was achieved through the use of 1x1 convolutions to perform dimensionality reduction before applying larger convolutions.

Conclusion::
This network was perhaps the pioneer in demonstrating a clear demarcation between the stem (initial data capture), the body (data interpretation), and the head (outcome prediction) within a CNN framework. 
This architectural blueprint has continued to be a mainstay in the construction of intricate networks to this day: the stem is composed of the initial two or three convolutional layers that process the image, distilling basic features from it. 
This is succeeded by a sequence of convolutional blocks that form the body. 

Architecture::
1. Follows a fixed number of Inception blocks
2. then, a global average pooling layer for classification

Inception block: 
         |-> 1x1 Convolution -> Relu -> |
input -> |-> 1x1 Convolution -> 3x3 Convolution -> Relu -> | -> Concatenate -> output
         |-> 1x1 Convolution -> 5x5 Convolution -> Relu -> |
         |-> 3x3 MaxPool -> 1x1 Convolution -> Relu -> |
"""

# Load the CIFAR10 data
batch_size = 64
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, input_shape = helper.load_dataset_image_data("CIFAR10" , batch_size, resize=(224, 224))
train_loader, valid_loader ,test_loader = data

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_1x1pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)
    
    
class GoogleLeNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(GoogleLeNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
model = GoogleLeNet(3, 10).to(device)
helper.model_summary(model, input_shape)


# Hyperparameters
learning_rate = 0.001
epochs = 1

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
trainer = helper.Trainer(model, optimizer, criterion, lr=learning_rate, max_epochs=epochs)

trainer.train(train_loader, valid_loader)

trainer.plot("NiN")

trainer.test(test_loader)
