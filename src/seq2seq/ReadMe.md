## Sequence to Sequence Learning
Sequence to Sequence learning is a technique used in Natural Language Processing to convert one sequence to another sequence. It is used in various tasks like Machine Translation, Text Summarization, Image Captioning, etc. The basic idea is to have Encoder and Decoder. Encoder reads the input sequence and converts it into a fixed-length context vector. Decoder reads the context vector and generates the output sequence.

### Reading List
- [*] [Pytorch Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq/tree/main)
- [ ] [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [ ] [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [ ] [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [ ] [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [ ] [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [ ] [The Transformer Family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)
- [ ] [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)

### Implementations
We will be implementing common Seq2Seq architectures from scratch using PyTorch for Machine Translation tasks.

- [x] [Simple Seq2Seq](/src/seq2seq/01_seq2seq.py)
    - Simple Seq2Seq model with RNN encoder and RNN decoder
    - Used teacher forcing to train the model

    ![Simple Seq2Seq](/assets/images/seq2seq1.png)
- [x] [Learning Phrase Representations](/src/seq2seq/02_seq2seq_learning_phrase_representations.py)
    - In Previous implementation, we used the last hidden state of the encoder as the initial hidden state of the decoder.
One of the major drawbacks of this approach is that decoder might not be able to use the information from the encoder effectively when the input sequence is long.
In this implementation, we will use encoder's last hidden state with input and output sequences. So, the decoder can use the information from the encoder effectively.

    ![Simple Seq2Seq](/assets/images/seq2seq7.png)
- [x] [NMT Jointly Learning to Align and Translate](/src/seq2seq/03_seq2seq_nmt_jointly_learning_to_align.py)
    - In previous implementation, we used the last hidden state of the encoder to concatenate with the decoder's hidden state to predict the next word. 
Instead of using the last hidden state of the encoder, we will use the attention mechanism to align the input and output sequences.

    ![Encoder](/assets/images/seq2seq9.png)
    ![Decoder](/assets/images/seq2seq10.png)
