import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import helper

"""
Introduction::
    In this file, we will implement VGG architecture for CIFAR10 dataset. 
    VGG was introduced by Karen Simonyan and Andrew Zisserman in 2014. 
    It was the runner-up architecture of ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014.

Motivation::
The key idea behind this paper was to understand whether deep or wide networks perform better.  
For instance, the successive application of two 3x3 convolutions touches the same pixels as a single 5x5 convolution does. 
At the same time, 3x3 use fewer parameters in comparison to 5x5 convolution.

Conclusion::
Authors found that the depth of the network is important for the performance of the network.

Architecture::
1. Follows a fixed number of VGG blocks
2. then, a fully connected layers for classification

VGG block: N * (CNN -> Relu) -> MaxPool

VGG Architecture: N * (VGG-Block) -> Fully Connected Layer -> Fully Connected Layer -> Output Layer

"""
batch_size = 64
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, input_shape = helper.load_dataset_image_data("CIFAR10" , batch_size)
train_loader, valid_loader ,test_loader = data


def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self,arch, in_channels, num_classes=10):
        super(VGG, self).__init__()
        conv_layers = []
        for (num_convs, out_channels) in arch:
            conv_layers.append(vgg_block(in_channels, out_channels, num_convs))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = VGG(arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)], in_channels=3, num_classes=10)
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

trainer.plot("VGG")

trainer.test(test_loader)
