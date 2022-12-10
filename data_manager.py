import torch
import torchvision
import torchvision.transforms as transforms
from transformations import Transformations
from torch.utils.data import Dataset
import os

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
    def __init__(self, batch_size, trainloader, valloader, testloader) -> None:
        self.batch_size = batch_size
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # Setup for question 9
        # elif dataset == "mnist" and transform == True:
        #     test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
        #     train_val = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        #     train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
        #     split_1, split_2, split_3, split_4, split_5, remain_train = torch.utils.data.random_split(train, [1000, 1000, 1000, 1000, 1000, 45000])

        #     augm_1 = DatasetFromSubset(subset=split_1, transform=Transformations.CenterCropSmall)
        #     augm_2 = DatasetFromSubset(subset=split_2, transform=Transformations.CenterCropLarge)
        #     augm_3 = DatasetFromSubset(subset=split_3, transform=Transformations.ColorJitter)
        #     augm_4 = DatasetFromSubset(subset=split_4, transform=Transformations.GaussianBlur)
        #     augm_5 = DatasetFromSubset(subset=split_5, transform=Transformations.RandomRotation)

        #     train = torch.utils.data.ConcatDataset([augm_1, augm_2, augm_3, augm_4, augm_5, train])

    @staticmethod
    def create_mnist(batch_size: int):
        train, val, test = DataManager.get_data_mnist('data', train_size=50000, val_size=10000)
        trainloader, valloader, testloader = DataManager.get_data_loaders(train, val, test, batch_size)
        return DataManager(batch_size, trainloader, valloader, testloader)

    @staticmethod
    def create_resized(batch_size: int):
        folder = os.path.join('data', 'mnist-varres')
        train, val, test = DataManager.get_data_resized(folder=folder, train_size=50000, val_size=10000)
        trainloader, valloader, testloader = DataManager.get_data_loaders(train, val, test, batch_size)
        return DataManager(batch_size, trainloader, valloader, testloader)

    @staticmethod
    def create_different_sizes(batch_size: int):
        sizes = ['small', 'med', 'large']
        train_sizes = [15931, 16064, 16006]
        val_sizes = [3982, 4016, 4001]
        train, val, test = DataManager.get_data_different_sizes('data', sizes, train_sizes, val_sizes)
        trainloader, valloader, testloader = DataManager.get_data_loaders(train, val, test, batch_size)
        return DataManager(batch_size, trainloader, valloader, testloader)

    @staticmethod
    def get_data_mnist(folder: str, train_size: int, val_size: int):
        train_val = torchvision.datasets.MNIST(root=folder, train=True, download=True, transform=transforms.ToTensor())
        train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
        test = torchvision.datasets.MNIST(root=folder, train=False, download=True, transform=transforms.ToTensor())

        return train, val, test

    @staticmethod
    def get_data_resized(folder: str, train_size: int, val_size: int):
        train_folder = os.path.join(folder, 'train')
        test_folder = os.path.join(folder, 'test')
        train_val = torchvision.datasets.ImageFolder(root=train_folder, transform=Transformations.Resize)
        test = torchvision.datasets.ImageFolder(root=test_folder, transform=Transformations.Resize)
        train, val = torch.utils.data.random_split(train_val, [train_size, val_size])

        return train, val, test

    @staticmethod
    def get_data_different_sizes(folder: str, sizes: list[str], train_sizes: list[str], val_sizes: list[str]):
        train_data = []
        val_data = []
        test_data = []
        for size, train_size, val_size in zip(sizes, train_sizes, val_sizes):
            size_folder_name = 'mnist-varres-' + size
            temp_train, temp_val, temp_test = DataManager.get_data_resized(os.path.join(folder, size_folder_name), train_size, val_size)
            train_data.append(temp_train)
            val_data.append(temp_val)
            test_data.append(temp_test)

        train = DataManager.concat_data(*train_data)
        val = DataManager.concat_data(*val_data)
        test = DataManager.concat_data(*test_data)

        return train, val, test

    @staticmethod
    def concat_data(*subsets):
        datasets = []
        for subset in subsets:
            if isinstance(subset, torch.utils.data.dataset.Subset):
                print(type(subset))
                datasets.append(DatasetFromSubset(subset))
                #datasets.append(torch.utils.data.Dataset(subset))
            else:
                datasets.append(subset)
        return torch.utils.data.ConcatDataset(datasets)

    @staticmethod
    def get_data_loaders(train, val, test, batch_size: int, shuffle = True, num_workers = 2):
        trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return trainloader, valloader, testloader
        


