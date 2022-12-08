import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from data_manager import DataManager
import pickle

class NetManager:
    def __init__(self, net: nn.Module, data_manager: DataManager , num_epochs: int = 2, learning_rate: float = 0.001) -> None:
        self.net = net
        self.data_manager = data_manager
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.test_accuracies = dict()

    def train(self, num_epochs: int, learning_rate: float):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.data_manager.trainloader, 0):
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
                if (i / self.data_manager.batch_size) % 2000 == 1999:    # print every 2000 samples
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

    def get_val_accuracy(self):
        return self.get_accuracy(data_loader=self.data_manager.valloader, print_results=False)

    def get_test_accuracy(self):
        self.test_accuracies = self.get_accuracy(data_loader=self.data_manager.testloader, print_results=True)
        return self.test_accuracies

    def get_accuracy(self, data_loader: torch.utils.data.DataLoader, print_results: bool = False) -> dict[int, float]:
        classes = range(10)
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)

        # again no gradients needed
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = self.net(inputs)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        accuracies = dict()

        # print accuracy for each class
        for classnumber, correct_count in correct_pred.items():
            accuracy = float(correct_count) / total_pred[classnumber]
            accuracies[classnumber] = accuracy
            if print_results:
                print(f'Accuracy for class: {str(classnumber)} is {accuracy:.1f} %')

    def save(self, filename: str = '', folder_path: str = 'saved'):
        path_model = os.path.join(folder_path, filename + '_model.pth')
        path_object = os.path.join(folder_path, filename + '_object.pkl')
        torch.save(self.net.state_dict(), path_model)
        with open(path_object, 'wb', -1) as fout:
            pickle.dump(self, fout, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str = '', folder_path: str = 'saved'):
        path_model = os.path.join(folder_path, filename + '_model.pth')
        path_object = os.path.join(folder_path, filename + '_object.pkl')
        self.net.load_state_dict(torch.load(path_model))
        with open(path_object, 'rb', -1) as fin:
            pickle.load(self, fin, pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # remove the network variable
        del state['net']
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        file = open(self.filename)
        for _ in range(self.lineno):
            file.readline()
        # Finally, save the file.
        self.file = file