import torch
import torchtext
import datasets
import spacy
import time

def load_translation_data(batch_size=64, tokenizer_type="spacy"):
    # 1. Load the dataset
    dataset = datasets.load_dataset("bentrevett/multi30k")
    train_data, valid_data, test_data = dataset['train'], dataset['validation'], dataset['test']

    # 2. Create the tokenizer
    en_nlp = spacy.load("en_core_web_sm")
    de_nlp = spacy.load("de_core_news_sm")
    def tokenize_data(data, en_nlp, de_nlp, max_len, sos_token, eos_token, lower=False):
        en_tokens = [token.text for token in en_nlp.tokenizer(data["en"])[:max_len]]
        de_tokens = [token.text for token in de_nlp.tokenizer(data["de"])[:max_len]]
        if lower:
            en_tokens = [token.lower() for token in en_tokens]
            de_tokens = [token.lower() for token in de_tokens]
        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        return {"en_tokens" : en_tokens, "de_tokens": de_tokens}

    # 3. Tokenize the data
    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    
    fn_kwargs = {"en_nlp": en_nlp, "de_nlp": de_nlp, "max_len": 1_000, "sos_token": sos_token, "eos_token": eos_token ,"lower": True}
    train_data = train_data.map(tokenize_data, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_data, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_data, fn_kwargs=fn_kwargs)

    # 4. Build the vocabulary
    min_freq = 2
    specials = [unk_token, pad_token, sos_token, eos_token]

    en_vocab = torchtext.vocab.build_vocab_from_iterator(train_data["en_tokens"], specials=specials, min_freq=min_freq)
    de_vocab = torchtext.vocab.build_vocab_from_iterator(train_data["de_tokens"], specials=specials, min_freq=min_freq)
    en_vocab.set_default_index(en_vocab[unk_token])
    de_vocab.set_default_index(de_vocab[unk_token])

    # 5. Numericalize the data
    def numericalize_data(data, en_vocab, de_vocab):
        en_indices = [en_vocab[token] for token in data["en_tokens"]]
        de_indices = [de_vocab[token] for token in data["de_tokens"]]
        return {"en_ids": en_indices, "de_ids": de_indices}

    train_data = train_data.map(numericalize_data, fn_kwargs={"en_vocab": en_vocab, "de_vocab": de_vocab})
    valid_data = valid_data.map(numericalize_data, fn_kwargs={"en_vocab": en_vocab, "de_vocab": de_vocab})
    test_data = test_data.map(numericalize_data, fn_kwargs={"en_vocab": en_vocab, "de_vocab": de_vocab})

    # 6. Convert the data to PyTorch format
    train_data = train_data.with_format(type="torch", columns=["en_ids", "de_ids"], output_all_columns=True)
    valid_data = valid_data.with_format(type="torch", columns=["en_ids", "de_ids"], output_all_columns=True)
    test_data = test_data.with_format(type="torch", columns=["en_ids", "de_ids"], output_all_columns=True)

    # 7. Create the DataLoader
    def collate_fn(batch):
        en_data = [item["en_ids"] for item in batch]
        de_data = [item["de_ids"] for item in batch]
        en_data = torch.nn.utils.rnn.pad_sequence(en_data, padding_value=en_vocab[pad_token])
        de_data = torch.nn.utils.rnn.pad_sequence(de_data, padding_value=de_vocab[pad_token])
        return {
            "en_ids": en_data,
            "de_ids": de_data
        }

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    print("\nData Summary\n" + "-" * 50)
    print(f"Train Data: {len(train_data)} | Valid Data: {len(valid_data)} | Test Data: {len(test_data)}")
    print(f"English Vocab Size: {len(en_vocab)} | German Vocab Size: {len(de_vocab)}")
    print(f"English Example Idx: {train_data[0]['en_ids']}")
    print(f"German Example Idx: {train_data[0]['de_ids']}")
    print("-" * 50)    
    return (train_loader, valid_loader, test_loader), (en_vocab, de_vocab)

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

    def train_epoch(self, train_loader, epoch=0, clip=0.1, teacher_forcing_ratio=0.5):
        self.model.train()
        total_acc, total_loss = 0, 0
        log_interval = 100
        start_time = time.time()
        total_loss = 0
        total_count = 0
        for inx, data in enumerate(train_loader):    
            src, trg = data["en_ids"].to(self.device), data["de_ids"].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, trg, teacher_forcing_ratio)
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            # output = [(trg len - 1) * batch size, output dim]
            output = output[1:].view(-1, output_dim) 
            # trg = [trg len, batch size]
            trg = trg[1:].view(-1)
            loss = self.criterion(output, trg)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            loss.backward()
            self.optimizer.step()
            total_acc += (output.argmax(1) == trg).sum().item()
            total_count += trg.size(0)
            total_loss += loss.item()
            if inx % log_interval == 0 and inx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f} | loss {:8.3f}".format(
                        epoch, inx, len(train_loader), total_acc / total_count, loss.item()
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()
        return total_loss / len(train_loader), total_acc / total_count

    def validate(self, val_loader):
        self.model.eval()
        total_acc, total_count = 0, 0
        total_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                src, trg = data["en_ids"].to(self.device), data["de_ids"].to(self.device)
                output = self.model(src, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = self.criterion(output, trg)
                total_acc += (output.argmax(1) == trg).sum().item()
                total_count += trg.size(0)
                total_loss += loss.item()
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
            for idx, data in enumerate(test_loader):
                src, trg = data["en_ids"].to(self.device), data["de_ids"].to(self.device)
                output = self.model(src, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = self.criterion(output, trg)
                total_acc += (output.argmax(1) == trg).sum().item()
                total_count += trg.size(0)
        print(f"Test Accuracy: {total_acc / total_count}")
        
    def save_model(self, name):
        torch.save(self.model.state_dict(), f"artifacts/models/{name}.pth")
        print(f"Model saved at artifacts/models/{name}.pth")


def model_summary(model):
    print("\nModel Summary\n" + "-" * 50)
    print(model)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6}M")
    print("-" * 50)
