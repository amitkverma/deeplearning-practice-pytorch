import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import helper

"""
In this file, we will implement Network in Network (NiN) architecture for CIFAR10 dataset. 
NiN was introduced by Min Lin, Qiang Chen, and Shuicheng Yan in 2013.

Motivation::
Previous architectures impose following limitations:
1. The fully connected layers at the end of the architecture consume tremendous numbers of parameters.
2. It is equally impossible to add fully connected layers earlier in the network to increase the degree of nonlinearity.

Conclusion::
NiN architecture is a good example of how 1x1 convolutional layers can be used to replace fully connected layers. 
Also, NiN architecture uses global average pooling layer to produce the output. Thus, it reduces the number of parameters in the network.


Architecture::
1. Follows a fixed number of NiN blocks
2. then, a global average pooling layer for classification

NiN block: Conv -> Relu -> Conv -> Relu -> Conv -> Relu

NiN Architecture: N * (NiN-Block) -> Global Average Pooling Layer -> Output Layer
"""

# Load the CIFAR10 data
batch_size = 64
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, input_shape = helper.load_dataset_image_data("CIFAR10" , batch_size, resize=(224, 224))
train_loader, valid_loader ,test_loader = data


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True)
    )
    
    
class NiN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(NiN, self).__init__()
        self.features = nn.Sequential(
            nin_block(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            nin_block(384, num_classes, kernel_size=3, stride=1, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
model = NiN(3, 10)
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
