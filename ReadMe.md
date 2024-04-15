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
        - [x] [RNN from scratch](/rnn/01_rnn_scratch.py)            
            - RNN provides way to capture sequential information using self-attention
        - [x] [RNN using Pytorch](/rnn/02_rnn_simple.py)
            - RNN has 1 hidden state and 1 output state
            - RNN takes input and hidden state as input and gives output and hidden state as output
            - Sequences are fed one by one to RNN and hidden state is passed to next sequence
            - Gradient clipping is used to avoid exploding gradients (nn.utils.clip_grad_norm_)
        - [x] [Deep RNN](/rnn/03_rnn_complex.py)
            - Added Multiple layers in RNN - Stacking RNN layers top of each other
            - Bidirectional RNN - Helps in capturing context from both directions
        - [x] [Optimized Training RNN](/rnn/04_rnn_tunned.py)
            - Gradient Clipping - Helps in avoiding exploding gradients
            - Weight Initialization - Helps model to converge faster
            - Dropout Regularization - Helps in avoiding overfitting
            - Pretrained Embeddings - Helps in learning better embeddings
            - Packed Sequence for variable length sequences - Helps in handling less padded sequences efficiently
    - [x] [Optimized Training LSTM](/rnn/05_lstm.py)
        - LSTM has 3 gates - Forget, Input, Output
        - LSTM has 2 states - Cell state, Hidden state
        - LSTM helps in capturing long term dependencies
        - LSTM helps in avoiding vanishing gradients
        - LSTM converges faster than RNN
    - [x] [GRU](/rnn/06_gru.py)
        - GRU has 2 gates - Reset, Update
        - GRU has 1 state - Hidden state
        - GRU helps in capturing long term dependencies
        - GRU helps in avoiding vanishing gradients
        - GRU converges faster than RNN
        - GRU and LSTM are similar in performance but GRU has less parameters
- [ ] [Sequence to Sequence](/seq2seq/)
    - [x] [Simple Seq2Seq](/seq2seq//01_seq2seq.py)
        - Simple Seq2Seq model with RNN encoder and RNN decoder
        - Used teacher forcing to train the model
    - [x] [Learning Phrase Representations](/seq2seq/02_seq2seq_learning_phrase_representations.py)
        - Using encoders hidden state concatenated with decoders hidden state and encode input  
    - [x] [NMT Jointly Learning to Align and Translate](/seq2seq/03_seq2seq_nmt_jointly_learning_to_align.py)
        - Using attention mechanism to align source and target sequences 
- [ ] [Transformer](/transformers/)
        - [x] [Introduction to HuggingFace Transformers](./transformers/01_introduction_to_transformers.py)
        - [x] [Inside of the HuggingFace pipeline API](./transformers/02_inside_of_pipeline_api.py)
        - [x] [Tokenizers in HuggingFace Transformers](./transformers/03_tokenizers_in_huggingface_transformers.py)
        - [x] [Embeddings in Transformers](./transformers/04_embeddings_in_transformers.py)
        - [x] [Fine-tuning a pre-trained model with HuggingFace Transformers](./transformers/05_finetune_transformers.py)
        - [ ] [Training a custom model with HuggingFace Transformers]()


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
