import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

class LoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(in_features=64*3*3, out_features=10)

    def forward(self, x):
        model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3,
            nn.Flatten(),
            self.fc
        )
        return model(x)

class LoNetHandler:
    def __init__(self, net: nn.Module) -> None:
        self.net = net
        self.trainloader, self.valloader, self.testloader = self.load_MNIST()

    def load_MNIST(self, batch_size: int = 5, train_size: int = 50000, val_size: int = 10000) -> tuple:
        train_val = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        train, val = torch.utils.data.random_split(train_val, [train_size, val_size])
        test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=2)

        return trainloader, valloader, testloader

    def train(self, num_epochs: int = 2, learning_rate: float = 0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

    def get_accuracy(self):
        classes = [str(n) for n in range(10)]
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)

        # again no gradients needed
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = self.net(inputs)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def save(self, folder_path: str = 'saved'):
        path = os.path.join(folder_path, 'mnist_model.pth')
        torch.save(self.net.state_dict(), path)

    def load(self, folder_path: str = 'saved'):
        path = os.path.join(folder_path, 'mnist_model.pth')
        self.net.load_state_dict(torch.load(path))

if __name__=='__main__':
    pass