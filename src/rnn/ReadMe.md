## Recurrent Neural Networks
Recurrent Neural Networks (RNNs) are a type of neural network that is designed to handle sequential data. They are used in a variety of applications, including natural language processing, speech recognition, and time series analysis. RNNs are particularly well-suited for tasks that involve processing sequences of data, such as text or audio.

### Reading List
- [x] [d2l.ai](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
- [x] [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [x] [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [x] [Understanding self-attention in RNN](https://distill.pub/2016/augmented-rnns/)
- [x] [Visualizing memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/)
- [x] [RNN Cheatsheet](./recurrent-neural-networks.pdf)
- [x] [Implementation of common models](https://github.com/bentrevett/pytorch-sentiment-analysis)

### Implementations
We will be implementing common RNN architectures from scratch using PyTorch for sentiment analysis tasks.

- [x] [RNN from scratch](/src/rnn/01_rnn_scratch.py)
    - RNN provides way to capture sequential information using self-attention

    ![RNN](/assets/images/rnn.svg)
- [x] [RNN using Pytorch](/src/rnn/02_rnn_simple.py)
    - RNN has 1 hidden state and 1 output state
    - RNN takes input and hidden state as input and gives output and hidden state as output
    - Sequences are fed one by one to RNN and hidden state is passed to next sequence
    - Gradient clipping is used to avoid exploding gradients (nn.utils.clip_grad_norm_)

    ![RNN](/assets/images/rnn-train.svg)

- [x] [Deep RNN](/src/rnn/03_rnn_complex.py)
    - Added Multiple layers in RNN - Stacking RNN layers top of each other
    - Bidirectional RNN - Helps in capturing context from both directions

    ![Deep RNN](/src/assets/images/deep-rnn.svg)
    ![Bi-RNN](/assets/images/birnn.svg)

- [x] [Optimized Training RNN](/src/rnn/04_rnn_tunned.py)
    - Gradient Clipping - Helps in avoiding exploding gradients
    - Weight Initialization - Helps model to converge faster
    - Dropout Regularization - Helps in avoiding overfitting
    - Pretrained Embeddings - Helps in learning better embeddings
    - Packed Sequence for variable length sequences - Helps in handling less padded sequences efficiently
    

- [x] [Optimized Training LSTM](/src/rnn/05_lstm.py)
    - LSTM has 3 gates - Forget, Input, Output
    - LSTM has 2 states - Cell state, Hidden state
    - LSTM helps in capturing long term dependencies
    - LSTM helps in avoiding vanishing gradients
    - LSTM converges faster than RNN

    ![LSTM](/assets/images/lstm.svg)

- [x] [GRU](/src/rnn/06_gru.py)
    - GRU has 2 gates - Reset, Update
    - GRU has 1 state - Hidden state
    - GRU helps in capturing long term dependencies
    - GRU helps in avoiding vanishing gradients
    - GRU converges faster than RNN
    - GRU and LSTM are similar in performance but GRU has less parameters

    ![GRU](/assets/images/gru.svg)


