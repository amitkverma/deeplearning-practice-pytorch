"""
In this file we will explore the tokenizers for transformers.
Since, Machine only understand numbers. We use Tokenization to converting a sequence of characters into a sequence of tokens (or Numbers). 
For example, the sentence "I love transformers" can be tokenized into ["I", "love", "transformers"].

There are three primary ways to tokenize:
## 1. Word-based Tokenization : As the name suggests, it involves breaking down text into individual words.
    e.g. "I love transformers" -> ["I", "love", "transformers"]
    **Pros:**
    * Easy to implement, especially for languages that use spaces.
    * The output is human-readable, which makes it easier for debugging and inspection.
    **Cons:**
    * Can't handle languages that don't use spaces consistently to separate words.
    * Inflexible with respect to out-of-vocabulary (OOV) words. If a word hasn't been seen before, the model might struggle with it.

## 2. Subword-based Tokenization : Splits text into subword units. It can help models understand morphemes or word parts.
    e.g. "unhappiness" -> ["un", "happi", "ness"]
    **Pros:**
    * Efficient in handling OOV words. Since words are broken down, even if the exact word hasn't been seen before, its parts might have been.
    * Can be more memory efficient because it reduces the vocabulary size. Instead of storing all forms of a word (like "happy", "happier", "happiest"), storing the parts ("happi", "er", "est") might suffice.
    **Cons:**
    * The output might not be as human-readable. This can make debugging or manual inspection slightly challenging.
    * Requires more sophisticated algorithms than simple space-based splits.

## 3. Character-based Tokenization : Splits text into individual characters.
    e.g. "I love transformers" -> ["I", " ", "l", "o", "v", "e", " ", "t", "r", "a", "n", "s", "f", "o", "r", "m", "e", "r", "s"]
    **Pros:**
    * The most flexible form of tokenization.
    * Can handle any language or text format.
    **Cons:**
    * Can be computationally expensive, especially for long texts.
    * The output is not human-readable, which can make debugging or manual inspection challenging.

Popular example of sub-word based tokenization algorithms include:
* Byte-Pair Encoding (BPE) - Used in models like GPT-2.
* WordPiece - Used in models like BERT.
* SentencePiece - Used in models like T5.
"""

text = "Tokenization is fun!"

# Word-based Tokenization
word_tokens = text.split()
print(f"Word tokens: {word_tokens}")

# A subword-based tokenization example
subword_tokens = [text[i:i+3] for i in range(0, len(text), 3)]
print(f"Subword tokens: {subword_tokens}")

# Character-based Tokenization
char_tokens = list(text)
print(f"Character tokens: {char_tokens}")


""" How to use Tokenizers in Transformers?

1. **Tokenizer**: The tokenizer is the main class that handles tokenization. It converts text into tokens that can be understood by the model.
2. **Token IDs**: Each token has a unique ID in the model's vocabulary. The tokenizer converts tokens into their respective IDs.
3. **Attention Mask**: This is a tensor that tells the model which tokens to pay attention to and which to ignore. It's useful for variable-length sequences.
4. **Special Tokens**: These are tokens that have special meanings. For example, [CLS] and [SEP] are used in BERT for classification and separation, respectively.
5. **Padding**: This is used to make all sequences the same length. It's necessary for efficient batch processing.
6. **Truncation**: This is used to cut sequences that are too long. It's necessary for efficient batch processing.
7. **Return Type**: The tokenizer can return tokens in different formats, like PyTorch tensors or TensorFlow tensors.

"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization is fun!"

# Tokenize the text
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {token_ids}")

# Combining the tokenization, conversion steps and other steps
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(f"Encoded input: {encoded_input}")

# The tokenizer returns a dictionary with the following keys
print(f"Token IDs: {encoded_input['input_ids']}")
print(f"Attention Mask: {encoded_input['attention_mask']}")
print(f"Token Type IDs: {encoded_input['token_type_ids']}")

# The tokenizer can also decode the token IDs back into text
decoded_text = tokenizer.decode(encoded_input['input_ids'][0])
print(f"Decoded text: {decoded_text}")