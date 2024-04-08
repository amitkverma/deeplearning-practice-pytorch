# Pytorch implementation of DeepLearning models
I have implemented these models from scratch for learning and understanding purposes. The models are implemented in a modular way, so that it can be easily understood and modified

## Models
- [x] [Basics of tensors](/basics)
    - [x] [Tensors Initialization](/basics/tensor_init.py)
    - [x] [Tensors Operations](/basics/tensor_operations.py)
    - [x] [Tensors Indexing](/basics/tensor_indexing.py)
    - [x] [Tensors Broadcasting](/basics/tensor_broadcasting.py)
- [ ] [Simple DNN model](/simple_dnn)
    - [ ] [Linear Regression]()
    - [ ] [Logistic Regression]()
    - [ ] [Multi Layer Perceptron]()
- [ ] [Convolutional Neural Network](/cnn)
    - [x] [LeNet](/cnn/2_LeNet.py)
        - LeNet provides a basic understanding of CNN
    - [x] [AlexNet](/cnn/3_AlexNet.py)
        - AlexNet provides a basic understanding of CNN stack which follows Convolution, Pooling, Activation, Fully Connected layers
    - [x] [VGG](/cnn/4_VGG.py)
        - VGG provides understanding deep or shallow network which is better
    - [x] [NiN](/cnn/5_NiN.py)
        - NiN provides understanding of Network in Network which helps in reducing number of parameters (introduces 1x1 convolutions)
    - [x] [GoogLeNet](/cnn/6_GoogLeNet.py)
        - GoogLeNet provides understanding of Inception module which is multi-branch architecture with the thought of stem, body and head architecture
    - [ ] [ResNet](/cnn/7_resnet.py)
        - ResNet provides understanding of Residual block which helps in avoiding vanishing gradients and helps in training deeper networks
- [x] [Recurrent Neural Network]() 
    - [x] [RNN](/rnn/)
        - [ ] [RNN from scratch](/rnn/1_RNN.py)            
            - RNN provides way to capture sequential information using self-attention
        - [x] [RNN using Pytorch](/rnn/rnn_simple.py)
            - RNN has 1 hidden state and 1 output state
            - RNN takes input and hidden state as input and gives output and hidden state as output
            - Sequences are fed one by one to RNN and hidden state is passed to next sequence
        - [x] [Deep RNN](/rnn/rnn_advance.py)
            - Added Multiple layers in RNN - Stacking RNN layers top of each other
            - Bidirectional RNN - Helps in capturing context from both directions
        - [x] [Optimized Training RNN](/rnn/rnn_advance.py)
            - Gradient Clipping - Helps in avoiding exploding gradients
            - Weight Initialization - Helps model to converge faster
            - Dropout Regularization - Helps in avoiding overfitting
            - Pretrained Embeddings - Helps in learning better embeddings
            - Packed Sequence for variable length sequences - Helps in handling less padded sequences efficiently
    - [x] [Optimized Training LSTM](/rnn/lstm_tuned.py)
        - LSTM has 3 gates - Forget, Input, Output
        - LSTM has 2 states - Cell state, Hidden state
        - LSTM helps in capturing long term dependencies
        - LSTM helps in avoiding vanishing gradients
        - LSTM converges faster than RNN
    - [x] [GRU](/rnn/gru_tuned.py)
        - GRU has 2 gates - Reset, Update
        - GRU has 1 state - Hidden state
        - GRU helps in capturing long term dependencies
        - GRU helps in avoiding vanishing gradients
        - GRU converges faster than RNN
        - GRU and LSTM are similar in performance but GRU has less parameters
- [ ] [Sequence to Sequence]()
    - [ ] [Simple Seq2Seq]()
    - [ ] [Attention Seq2Seq]()
- [ ] [Transformer]()
    - [ ] [BERT]()
    - [ ] [GPT]()

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python <model_name>.py
```

## References
- [Dive into Deep Learning](https://d2l.ai/)
- [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
