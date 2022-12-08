import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_manager import DataManager

class NetManager:
    def __init__(self, net: nn.Module, data_manager: DataManager , num_epochs: int = 2, learning_rate: float = 0.001) -> None:
        self.net = net
        self.data_manager = data_manager
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        self.losses = []
        self.val_accuracies = []
        self.test_accuracy = 0.0
        self.test_class_accuracies = dict()

    def train(self, print_results: bool = False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        self.losses = []

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            epoch_loss, val_acc = self._single_epoch()
            self.losses.append(epoch_loss)
            self.val_accuracies.append(val_acc)

            if print_results:
                print(f'Average loss of epoch {epoch + 1}: {epoch_loss}')
                print(f'Validation accuracy of epoch {epoch + 1}: {val_acc * 100:.1f} %')
        if print_results:
            print('Finished Training')

    def _single_epoch(self, device: torch.device):
        epoch_loss = 0.0
        for data in self.data_manager.trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(self.data_manager.trainloader)
        val_acc = self._get_accuracy(self.data_manager.valloader)
        return epoch_loss, val_acc

    def test(self, print_results: str = False) -> tuple[float, dict[int, float]]:
        self.test_accuracy = self.get_accuracy(self.data_manager.testloader)
        self.test_class_accuracies = self.get_class_accuracy(self.data_manager.testloader)

        if print_results:
            print(f'Total test accuracy: {self.test_accuracy * 100:.1f} %')
            for k, v in self.test_class_accuracies.items():
                print(f'Accuracy of class {str(k)}: {v * 100:.1f} %')

    def save_model(self, folder_path: str = 'saved'):
        path = os.path.join(folder_path, 'model.pth')
        torch.save(self.net.state_dict(), path)

    def load_model(self, folder_path: str = 'saved'):
        path = os.path.join(folder_path, 'model.pth')
        self.net.load_state_dict(torch.load(path))

    def _get_class_accuracy(self, data_loader: torch.utils.data.DataLoader) -> dict[int, float]:
        accuracies = dict()
        correct_pred, total_pred = self.get_predictions(data_loader)

        # print accuracy for each class
        for classnumber, correct_count in correct_pred.items():
            accuracy = float(correct_count) / total_pred[classnumber]
            accuracies[classnumber] = accuracy

        return accuracies

    def _get_accuracy(self, data_loader: torch.utils.data.DataLoader) -> float:
        correct_pred, total_pred = self.get_predictions(data_loader)

        total_accuracy = sum(correct_pred.values()) / sum(total_pred.values())
        return total_accuracy

    def _get_predictions(self, data_loader: torch.utils.data.DataLoader) -> tuple[dict, dict]:
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

        return correct_pred, total_pred
