import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def load_mnist_data(batch_size: int) -> Tuple[DataLoader]:
    """Loads MNIST data and returns train and test loaders.
    
    Args:
        batch_size (int): batch size
    
    Returns:
        Tuple[DataLoader]: train and test loaders
    """
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader
