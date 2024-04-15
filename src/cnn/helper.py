import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets

def load_dataset_image_data(data_type, batch_size, resize=None, val_split=0.25):
    # Define transformations: Resize if needed, then convert images to tensors
    if resize:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()

    # Dynamically get the dataset class from torchvision.datasets
    dataset_class = getattr(datasets, data_type)

    # Load the dataset for training (without separating train and test yet)
    full_train_dataset = dataset_class(root='data/', train=True, transform=transform, download=True)

    # Determine sizes for training and validation sets
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Load the test dataset
    test_dataset = dataset_class(root='data/', train=False, transform=transform, download=True)
    

    # Create data loaders for the datasets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset sizes
    print("---------- Dataset Summary ----------")
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))
    print("Test dataset size: ", len(test_dataset))
    
    # Retrieve a single batch to check the data shape
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print("Data shape (images): ", images.shape)  # Example shape: torch.Size([batch_size, channels, height, width])
    print("Labels shape: ", labels.shape)  # Example shape: torch.Size([batch_size])
    print("-------------------------------------")

    return (train_loader, val_loader, test_loader), images.shape


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, lr=0.001, max_epochs=5):
        self.device = self.get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer else torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.lr = lr
        self.max_epochs = max_epochs
        self.train_loss = []
        self.val_loss = []
        
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() > 0 else "cpu")
    
    def train_epoch(self, train_loader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model if a validation loader is provided."""
        if not val_loader:
            return None
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader=None):
        """Run the training loop."""
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else 'Not Available'
            self.val_loss.append(val_loss)
            self.train_loss.append(train_loss)
            print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss}")
            
    def plot(self, name, show=False):
        import matplotlib.pyplot as plt
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'assets/losses/{name}_loss.png')
        if show:
            plt.show()
        
    def test(self, test_loader):        
        num_correct = 0
        num_samples = 0
        self.model.eval()
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                
                _, predictions = output.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


def model_summary(model, input_size, device = 'cpu'):
    from torchsummary import summary
    batch_size = input_size[0]
    input_size = input_size[1:]
    summary(model, input_size, batch_size, device)