from transformers import pipeline

"""
!pip install transformers
"""

"""
In this file, we will use basic nlp pipeline from transformers library to explore the capabilities of the library.
Hugging Face provides a wide range of pre-trained models and pipelines for various NLP tasks. 
For example:
- feature-extraction (get the vector representation of a text)
- fill-mask
- ner (named entity recognition)
- question-answering
- sentiment-analysis
- summarization
- text-generation
- translation
- zero-shot-classification
"""

# To use a pipeline, we can use the pipeline function and specify the task we want to perform.
# For example, to perform sentiment analysis, we can use the following code:
classifier = pipeline("sentiment-analysis")
out = classifier("I love transformers") # [{'label': 'POSITIVE', 'score': 0.9995483756065369}]
print(out)

# Another example is to use the fill-mask pipeline to fill in the blanks in a sentence.
fill_mask = pipeline("fill-mask")
out = fill_mask("I <mask> transformers")
# [{'score': 0.05126520246267319, 'token': 10003, 'token_str': 'onic', 'sequence': 'Ionic transformers'}, ...]
print(out)

# Note: 
# The pipeline function downloads the model and tokenizer from the Hugging Face model hub.
# The model is cached in the local directory, so it will not be downloaded again when the function is called next time.
