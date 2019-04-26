import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import pdb
import os
import time

class ProjNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ProjNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.full = nn.Linear(576, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 576)
        x = self.full(x)
        return x

class Project2:
    
    def __init__(
        self,
        use_transform=True,
        logs_path='./logs/',
        data_path='./data/',
        clean_logs=True
    ):
        self.logs_path = logs_path
        self.clean_logs=clean_logs
        self.data_path=data_path
    
    #################################################################################
    
    def build_data_loaders(self, dataset='cifar', batch_size=128, use_transform=True):
        if use_transform:
            transform = transforms.Compose(
                [transforms.ToTensor(), 
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            
        if dataset == 'cifar':
            train_set = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
            test_set = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        elif dataset == 'mnist':
            train_set = torchvision.datasets.MNIST(root=self.data_path, train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
            test_set = torchvision.datasets.MNIST(root=self.data_path, train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            raise Exception('UNKNOWN DATASET')
            
        self.data_loaders = {
            'train_set': train_set,
            'test_set': test_set,
            'trainloader': trainloader,
            'testloader': testloader
        }
        
        print('status: data preparation is complete.\n')
        
        
    #################################################################################
    
    def build_model(self, target_dataset='cifar'):
        if target_dataset == 'cifar':
            self.net = ProjNet(in_channels=3)
        elif target_dataset == 'mnist':
            self.net = ProjNet(in_channels=1)
        else:
            raise Exception('UNKNOWN DATASET')
        print('status: the model is built.\n')
    
    #################################################################################
    def build_optimizers(self):
        self.optimizers = {
            'sgd': optim.SGD(self.net.parameters(), lr=0.01, momentum=0),
            'sgd_with_momentum': optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9),
            'adadelta': optim.Adadelta(self.net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0),
            'adagrad': optim.Adagrad(self.net.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0),
            'adam': optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False),
            'adamax': optim.Adamax(self.net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
            'asgd': optim.ASGD(self.net.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0),
            'rmsprop': optim.RMSprop(self.net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False),
            'rprop': optim.Rprop(self.net.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        }
        print('optimizers are built.\n')
        
    def perform_an_experiment(
        self,
        dataset='cifar',
        title='experiment_1',
        optimizer='sgd',
        use_transform=True,
        max_number_of_epochs=50
    ):
        # re-initializing the model
        print('status: {} in progress...\n\n'.format(title))
        #os.system('rm ./runs/*')
        self.net = None
        self.optimizers = None
        criterion = nn.MSELoss()
        
        self.build_data_loaders(dataset=dataset, use_transform=use_transform)
        if use_transform:
            transform_status = 'withTransform'
        else:
            transform_status = 'withoutTransform'
        
        train_set = self.data_loaders['train_set']
        test_set = self.data_loaders['test_set']
        trainloader = self.data_loaders['trainloader']
        testloader = self.data_loaders['testloader']
        
        self.build_model(target_dataset=dataset)
        self.build_optimizers()
        
        writer = SummaryWriter()
        
        for epoch in range(max_number_of_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                y_onehot = labels.numpy()
                y_onehot = (np.arange(10) == y_onehot[:,None]).astype(np.float32)
                y_onehot = torch.from_numpy(y_onehot)
                self.optimizers[optimizer].zero_grad()
                
                outputs = self.net(inputs)
                loss = criterion(outputs, y_onehot)
                loss.backward()
                self.optimizers[optimizer].step()
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    writer.add_scalars(title + '/logs/loss_info', 
                                          {'running_loss': running_loss, 'epoch': epoch + 1, 'i': i + 1}
                                      )
                    running_loss = 0.0
                    #pdb.set_trace()
                
        test_accuracy = self.evaluate_on_dataset(testloader)
        print('test accuracy is {:2f}\n'.format(test_accuracy))

        if os.path.isfile('./test_accuracies.pkl'):
            temp = pickle.load(open('./test_accuracies.pkl', 'rb'))
            temp[title] = test_accuracy
            pickle.dump(temp, open('./test_accuracies.pkl', 'wb'))
        else:
            temp = {}
            temp[title] = test_accuracy
            pickle.dump(temp, open('./test_accuracies.pkl', 'wb'))

        writer.add_scalar(title + '/logs/test_accuracy', test_accuracy)
        writer.export_scalars_to_json("./all_scalars_{}_.json".format(title))
        writer.close()
        print('status: experiment {} is now complete and the results are saved.\n\n'.format(title))
        
    def perform_experiments(self):
        datasets = ['mnist', 'cifar']
        optimizers = ['sgd', 'sgd_with_momentum', 'adadelta', 'adagrad', 'adam', 'adamax', 'asgd', 'rmsprop', 'rprop']
        transform_usages = [True, False]
        for dataset in datasets:
            for optimizer in optimizers:
                for transform_usage in transform_usages:
                    if transform_usage:
                        transform_status = 'withTransform'
                    else:
                        transform_status = 'withoutTransform'
                    self.perform_an_experiment(
                        title='{}_{}_{}_experiment'.format(dataset, optimizer, transform_status),
                        dataset=dataset,
                        optimizer=optimizer,
                        use_transform=transform_usage
                    )
    def evaluate_on_dataset(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return float(100 * correct) / float(total)


if __name__=='__main__':
    project2 = Project2()
    project2.perform_experiments()

