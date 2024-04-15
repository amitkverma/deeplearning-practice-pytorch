# Pytorch implementation of DeepLearning models
This repository features various deep learning models implemented in PyTorch, both from scratch and with built-in modules. Each model is in a separate file for easy understanding and modification. It serves as a learning resource and reference for developing deep learning models with PyTorch.

## Models
- [x] [Basics of tensors](/src/basics)
    - [x] [Tensors Initialization](/src/basics/tensor_init.py)
    - [x] [Tensors Operations](/src/basics/tensor_operations.py)
    - [x] [Tensors Indexing](/src/basics/tensor_indexing.py)
    - [x] [Tensors Broadcasting](/src/basics/tensor_broadcasting.py)
- [ ] [Simple DNN model](/src/simple_dnn)
    - [ ] [Linear Regression]()
    - [ ] [Logistic Regression]()
    - [ ] [Multi Layer Perceptron]()
- [ ] [Convolutional Neural Network](/src/cnn)
    - [x] [LeNet](/src/cnn/2_LeNet.py)
    - [x] [AlexNet](/src/cnn/3_AlexNet.py)
    - [x] [VGG](/src/cnn/4_VGG.py)
    - [x] [NiN](/src/cnn/5_NiN.py)
    - [x] [GoogLeNet](/src/cnn/6_GoogLeNet.py)
    - [ ] [ResNet](/src/cnn/7_resnet.py)
    - [ ] [EefficientNet]()
    - [ ] [Transfer Learning and Fine-tuning]()
- [x] [Recurrent Neural Network](/src/rnn/) 
    - [x] [RNN from scratch](/src/rnn/01_rnn_scratch.py)            
    - [x] [RNN using Pytorch](/src/rnn/02_rnn_simple.py)
    - [x] [Deep RNN](/src/rnn/03_rnn_complex.py)
    - [x] [Optimized Training RNN](/src/rnn/04_rnn_tunned.py)
    - [x] [Optimized Training LSTM](/src/rnn/05_lstm.py)
    - [x] [GRU](/src/rnn/06_gru.py)
- [ ] [Sequence to Sequence](/src/seq2seq/)
    - [x] [Simple Seq2Seq](/src/seq2seq//01_seq2seq.py)
        - Simple Seq2Seq model with RNN encoder and RNN decoder
        - Used teacher forcing to train the model
    - [x] [Learning Phrase Representations](/src/seq2seq/02_seq2seq_learning_phrase_representations.py)
        - Using encoders hidden state concatenated with decoders hidden state and encode input  
    - [x] [NMT Jointly Learning to Align and Translate](/src/seq2seq/03_seq2seq_nmt_jointly_learning_to_align.py)
        - Using attention mechanism to align source and target sequences 
- [ ] [Transformer](/src/transformers/)
    - [x] [Introduction to HuggingFace Transformers](/src/transformers/01_introduction_to_transformers.py)
    - [x] [Inside of the HuggingFace pipeline API](/src/transformers/02_inside_of_pipeline_api.py)
    - [x] [Tokenizers in HuggingFace Transformers](/src/transformers/03_tokenizers_in_huggingface_transformers.py)
    - [x] [Embeddings in Transformers](/src/transformers/04_embeddings_in_transformers.py)
    - [x] [Fine-tuning a pre-trained model with HuggingFace Transformers](/src/transformers/05_finetune_transformers.py)
    - [ ] [Training a custom model with HuggingFace Transformers]()
- [ ] [Dataset Creation]()
    - [ ] [Custom Image Dataset in pytorch]()
    - [ ] [Custom text Dataset in pytorch]()
    - [ ] [Transformers datasets in pytorch]()
- [ ] [Techniques]
    - [ ] [Learning Rate finder]()
    - [ ] [Early Stopping]()
    - [ ] [Model Checkpointing]()
    - [ ] [Gradient Clipping]()
    - [ ] [Weight Initialization]()
    - [ ] [Dropout Regularization]()
    - [ ] [Batch Normalization]()
    - [ ] [Data Augmentation]()
    - [ ] [Transfer Learning]()
    - [ ] [Fine-tuning]()    
- [ ] [Others]()
    - [ ] [Saving and Loading saved models]()
    - [ ] [Visualizing the model using tensorboard]()



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
