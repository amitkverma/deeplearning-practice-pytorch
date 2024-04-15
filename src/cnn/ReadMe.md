## Convolutional Neural Networks
This chapter covers convolutional neural networks (CNNs), a type of neural network crucial for image analysis, introduced by LeCun et al. in 1995. Referred to as Convnets, these networks have become central in computer vision, NLP and other domain

### Reading List
- [x] [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [x] [cs231n](https://cs231n.github.io/convolutional-networks/)
- [x] [CNN Features Visualization](https://distill.pub/2017/feature-visualization/)
- [x] [Visualizing and Understanding Convolutional Networks](https://github.com/poloclub/cnn-explainer)
- [x] [CNN Cheatsheet](./convolutional-neural-networks.pdf)
- [x] [Implementation of common models](https://github.com/bentrevett/pytorch-image-classification)

### Implementations
We will be implementing common CNN architectures from scratch using PyTorch for **image classification** tasks.

- [x] [LeNet](/src/cnn/01_lenet.py)
    - LeNet provides a basic understanding of CNN

    ![](/assets/images/lenet.png)

- [x] [AlexNet](/src/cnn/3_AlexNet.py)
    - AlexNet provides a basic understanding of CNN stack which follows Convolution, Pooling, Activation, Fully Connected layers

    ![](/assets/images/alexnet.svg)
- [x] [VGG](/src/cnn/4_VGG.py)
    - VGG provides understanding deep or shallow network which is better

    ![](/assets/images/vgg.svg)
- [x] [NiN](/src/cnn/5_NiN.py)
    - NiN provides understanding of Network in Network which helps in reducing number of parameters (introduces 1x1 convolutions)

    ![NiN](/assets/images/nin.svg)
- [x] [GoogLeNet](/src/cnn/6_GoogLeNet.py)
    - GoogLeNet provides understanding of Inception module which is multi-branch architecture with the thought of stem, body and head architecture

    ![GoogLeNet](/assets/images/inception.svg)
- [ ] [ResNet](/src/cnn/7_ResNet.py)
    - ResNet provides understanding of Residual block which helps in avoiding vanishing gradients and helps in training deeper networks

    ![ResNet](/assets/images/residual-block.svg)

- [ ] [EefficientNet]()
- [ ] [Transfer Learning and Fine-tuning]()

