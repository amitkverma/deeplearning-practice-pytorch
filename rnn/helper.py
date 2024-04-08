import torch
import torchtext.datasets as datasets
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import time


###
# Process the text data
# ---------------------
# Download data -> Tokenize -> Build Vocab -> Numericalize -> Create DataLoader



def load_dataset_text_data(dataset_name, batch_size=64, tokenizer_type="basic_english", use_pretrained_vectors=False, use_padded_sequence=False):
    device = get_device()
    train_dataset, test_dataset = datasets.DATASETS[dataset_name]()
    train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)
    
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )

    tokenizer = get_tokenizer(tokenizer_type)

    # Tokenizer
    def tokenize_dataset(dataset):
        for _, text in dataset:
            yield tokenizer(text)
    
    # Build the vocabulary
    vocab = build_vocab_from_iterator(tokenize_dataset(train_dataset), min_freq=1, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    
    if use_pretrained_vectors:
        from torchtext.vocab import GloVe
        vocab.load_vectors(GloVe(name='6B', dim=100))

    if use_padded_sequence:
        def collate_batch(batch):
            Y, X = list(zip(*batch))
            Y = torch.tensor(Y) - 1
            X = [vocab(tokenizer(text)) for text in X]
            X = [torch.tensor(tokens) for tokens in X]
            X = pad_sequence(X, batch_first=True)
            return X.to(device), Y.to(device)
    else:
        def collate_batch(batch):
            max_word_length = 50
            Y, X = list(zip(*batch))
            Y = torch.tensor(Y) - 1
            X = [vocab(tokenizer(text)) for text in X]
            X = [tokens + [0]* (max_word_length - len(tokens)) if len(tokens) < max_word_length else tokens[:max_word_length] for tokens in X]
            return torch.tensor(X, dtype=torch.int32).to(device), Y.to(device)
        
    train_loader = DataLoader(
        split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    valid_loader = DataLoader(
        split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    num_class = len(set([label for (label, text) in train_dataset]))
    print("\nData Summary\n" + "-" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Number of classes: {num_class}")
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Train Dataset Size: {len(split_train_)}")
    print(f"Validation Dataset Size: {len(split_valid_)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    print(f"Batch Size: {batch_size}")
    print(f"Tokenizer: {tokenizer_type}")
    print(f"Input data shape: {next(iter(train_loader))[1].shape}")
    print(f"Output data shape: {next(iter(train_loader))[0].shape}")
    print(f"-" * 50)
    return (train_loader, valid_loader, test_loader), num_class, vocab
    

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, lr=0.001, max_epochs=5):
        self.device = self.get_device()
        self.model = model.to(self.device)
        self.optimizer = (
            optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=lr)
        )
        self.criterion = criterion if criterion else torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.max_epochs = max_epochs
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() > 0 else "cpu")

    def train_epoch(self, dataloader, epoch=0):
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        total_loss = 0
        for idx, (text, label) in enumerate(dataloader):
            predicted_label = self.model(text)
            loss = self.criterion(predicted_label, label)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f} | loss {:8.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count, loss.item()
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()
        return total_loss / len(dataloader), total_acc / total_count

    def validate(self, val_loader):
        """Validate the model if a validation loader is provided."""
        self.model.eval()
        total_acc, total_count = 0, 0
        total_loss = 0
        with torch.no_grad():
            for idx, (text, label) in enumerate(val_loader):
                predicted_label = self.model(text)
                loss = self.criterion(predicted_label, label)
                total_loss += loss.item()
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_loss / len(val_loader), total_acc / total_count

    def train(self, train_loader, val_loader=None):
        """Run the training loop."""
        print("\nTraining Started\n" + "-" * 50)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)
        total_accu = None
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, accu_val = (
                self.validate(val_loader) if val_loader else "Not Available"
            )
            self.val_acc.append(accu_val)
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            if total_accu is not None and total_accu > accu_val:
                scheduler.step()
            else:
                total_accu = accu_val
            print("-" * 59)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | "
                "valid accuracy {:8.3f} ".format(
                    epoch, time.time() - epoch_start_time, accu_val
                )
            )
            print("-" * 59)

    def plot(self, name, show=False):
        import matplotlib.pyplot as plt

        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.title("Training and Validation Loss")
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"artifacts/charts/{name}_loss.png")
        if show:
            plt.show()

    def test(self, test_loader):
        """Validate the model if a validation loader is provided."""
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (text, label) in enumerate(test_loader):
                predicted_label = self.model(text)
                loss = self.criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        print(f"Test Accuracy: {total_acc / total_count}")
        
    def save_model(self, name):
        torch.save(self.model.state_dict(), f"artifacts/models/{name}.pth")
        print(f"Model saved at artifacts/models/{name}.pth")


def model_summary(model):
    print("\nModel Summary\n" + "-" * 50)
    print(model)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6}M")
    print("-" * 50)
