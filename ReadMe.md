# Pytorch implementation of DeepLearning models
I have implemented these models from scratch for learning and understanding purposes. The models are implemented in a modular way, so that it can be easily understood and modified

## Models
- [x] [Basics of tensors](/basics)
- [x] [Simple DNN model](/simple_dnn)
    - [x] [Linear Regression]()
    - [x] [Logistic Regression]()
    - [x] [Multi Layer Perceptron]()
- [ ] [Convolutional Neural Network]()
    - [x] [LeNet](/cnn/2_LeNet.py)
    - [x] [AlexNet](/cnn/3_AlexNet.py)
    - [x] [VGG](/cnn/4_VGG.py)
    - [x] [NiN](/cnn/5_NiN.py)
    - [ ] [GoogLeNet](/cnn/6_GoogLeNet.py)
    - [ ] [ResNet](/cnn/7_resnet.py)
    - [ ] [DenseNet](/cnn/8_densenet.py)
- [ ] [Recurrent Neural Network]() 
    - [x] [RNN]()
            - [ ] [RNN from scratch](/rnn/1_RNN.py)
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
            - [x] [LSTM using Pytorch]()
                    - LSTM has 3 gates - Forget, Input, Output
                    - LSTM has 2 states - Cell state, Hidden state
                    - LSTM helps in capturing long term dependencies
                    - LSTM helps in avoiding vanishing gradients
                    - LSTM converges faster than RNN
    - [ ] [LSTM]()
    - [ ] [GRU]()
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
