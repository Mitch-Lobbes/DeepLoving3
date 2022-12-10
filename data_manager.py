import torch
import torchvision
import torchvision.transforms as transforms
from transformations import transformation
from transformations import Transformations
from torch.utils.data import Dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DataManager:
    def __init__(self, dataset: str, batch_size: int, train_size: int = 50000, val_size: int = 10000, transform=False) -> None:
        if dataset == 'mnist' and transform == False:
            train_val = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
            train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
            test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

        # Setup for question 9
        elif dataset == "mnist" and transform == True:
            test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
            train_val = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
            train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
            split_1, split_2, split_3, split_4, split_5, remain_train = torch.utils.data.random_split(train, [1000, 1000, 1000, 1000, 1000, 45000])

            augm_1 = DatasetFromSubset(subset=split_1, transform=Transformations.CenterCropSmall)
            augm_2 = DatasetFromSubset(subset=split_2, transform=Transformations.CenterCropLarge)
            augm_3 = DatasetFromSubset(subset=split_3, transform=Transformations.ColorJitter)
            augm_4 = DatasetFromSubset(subset=split_4, transform=Transformations.GaussianBlur)
            augm_5 = DatasetFromSubset(subset=split_5, transform=Transformations.RandomRotation)

            train = torch.utils.data.ConcatDataset([augm_1, augm_2, augm_3, augm_4, augm_5, train])
        
        # Setup for question 10
        elif dataset == "mnist-varres":
            # all_data = torchvision.datasets.ImageFolder(root='data', transform=Transformations.Resize())
            train_val = torchvision.datasets.ImageFolder(root='./data/mnist-varres/train', transform=Transformations.Resize)
            test = torchvision.datasets.ImageFolder(root='./data/mnist-varres/test', transform=Transformations.Resize)
            train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
        
        else:
            raise Exception("This setup for the DataManager is not supported.")

        self.batch_size = batch_size
        self.trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=2)