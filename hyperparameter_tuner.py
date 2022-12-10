import pandas as pd
from conv_net import ConvNet, ConvNetVariable
from data_manager import DataManager
from net_manager import NetManager

class Tuner:
    def __init__(self, num_epochs: int, batch_sizes: list[int], learning_rates: list[float], 
            betas_1: list[float], betas_2: list[float], weight_decays: list[float], net_type='fixed') -> None:
        self.num_epochs = num_epochs
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.betas_1 = betas_1
        self.betas_2 = betas_2
        self.weight_decays = weight_decays

        self.net_type = net_type

        col_names = ['batch_size', 'learning_rate', 'beta_1', 'beta_2', 'weight_decay', 
            'epoch', 'train_loss', 'train_accuracy', 'val_accuracy']
        self.df = pd.DataFrame(columns=col_names)

    def run(self) -> pd.DataFrame:
        for batch_size in self.batch_sizes:
            if self.net_type == 'fixed':
                data_managers = [DataManager.create_mnist(batch_size)]
            elif self.net_type == 'variable':
                data_managers = DataManager.create_different_sizes(batch_size)

            for learning_rate in self.learning_rates:
                for beta_1 in self.betas_1:
                    for beta_2 in self.betas_2:
                        for weight_decay in self.weight_decays:
                            self._run_single_configuration(batch_size, data_managers, self.num_epochs, 
                                learning_rate, beta_1, beta_2, weight_decay)

        return self.df

    def _run_single_configuration(self, batch_size, data_managers, num_epochs, learning_rate, beta_1, beta_2, weight_decay) -> None:
        if self.net_type == 'fixed':
            net = ConvNet()
        elif self.net_type == 'variable':
            net = ConvNetVariable(N=81, pool_type='avg')
        net_manager = NetManager(net, data_managers, num_epochs, 
            learning_rate, (beta_1, beta_2), weight_decay)
        net_manager.train()
        for epoch in range(num_epochs):
            row = pd.Series({'batch_size': batch_size, 'learning_rate': learning_rate, 
                'beta_1': beta_1, 'beta_2': beta_2, 'weight_decay': weight_decay, 'epoch': epoch, 
                'train_loss': net_manager.losses[epoch], 
                'train_accuracy':net_manager.train_accuracies[epoch], 
                'val_accuracy': net_manager.val_accuracies[epoch]})
            self.df = self.df.append(row, ignore_index=True)
