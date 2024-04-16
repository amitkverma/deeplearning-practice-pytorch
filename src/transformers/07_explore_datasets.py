"""
In this file we will explore the datasets library from the Hugging Face Transformers library.
"""

#  Load custom data using the datasets library
from datasets import load_dataset

wine_quality_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

wine_dataset = load_dataset("csv", data_files=wine_quality_dataset, sep=";") # we can also load json, text, etc.
print(wine_dataset)
print(wine_dataset["train"].features) # Check the features of the dataset
print(wine_dataset["train"][0]) # Peek at the first training example

#  Create train and test datasets if not already available
wine_dataset = wine_dataset["train"].train_test_split(test_size=0.2)
print(wine_dataset)
train_dataset = wine_dataset["train"]

# Play around with the dataset

# Select a subset of the dataset
subset_dataset = train_dataset.select(range(5))

# Filter the dataset
filtered_dataset = train_dataset.filter(lambda example: example["quality"] > 6)

# Shuffle the dataset
shuffled_dataset = train_dataset.shuffle(seed=42)

# Sort the dataset
sorted_dataset = train_dataset.sort("quality")

# Map a function to the dataset
def label_to_text(example):
    example["quality"] = "good" if example["quality"] >= 6 else "bad"
    return example

mapped_dataset = train_dataset.map(label_to_text)

# Create a new column in the dataset
train_dataset = train_dataset.map(lambda example: {"is_wine_good": example["quality"] == "good"})

# Parallelize the dataset

def parallelize_function(example):
    return example
train_dataset = train_dataset.map(parallelize_function, batched=True)

# Convert the dataset to a Pandas DataFrame
train_dataset.set_format("pandas")

# saving the dataset
train_dataset.save_to_disk("wine_quality_dataset")