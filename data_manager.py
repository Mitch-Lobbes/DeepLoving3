import torch
import torchvision
import torchvision.transforms as transforms

class DataManager:
    def __init__(self, dataset: str, batch_size: int, train_size: int, val_size: int) -> None:
        if dataset == 'mnist':
            train_val = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
            train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
            test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

        self.batch_size = batch_size
        self.trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=2)