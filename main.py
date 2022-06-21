import pickle
import numpy as np
import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
from skorch import NeuralNetClassifier
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(32),
    transforms.CenterCrop(32),
])


def import_datasets():
    """Import datasets"""
    dataset1 = torchvision.datasets.ImageFolder(root='./data/train/', transform=transform)
    dataset2 = torchvision.datasets.ImageFolder(root='./data/test/', transform=transform)
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

    """Random split data"""
    m = len(dataset)
    train, test = random_split(dataset, [m - int(m / 4), int(m / 4)])
    return dataset, train, test


if __name__ == '__main__':
    print('COMP 472 Project')
    print('AI Face Mask Detector\n')

    # Hyper-parameters
    num_epochs = 7
    num_classes = 4
    learning_rate = 0.001
    classes = ('cloth', 'n95', 'none', 'surgical')

    """Import training dataset"""
    _, train_dataset, _ = import_datasets()

    """Check device to train"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    """Start training"""
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        CNN.CNN(),
        max_epochs=num_epochs,
        lr=learning_rate,
        batch_size=100,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=device
    )
    y_train = np.array([np.int64(y) for x, y in iter(train_dataset)])
    net.fit(train_dataset, y=y_train)

    """Save model state_dict"""
    with open('model-pkl.pkl', 'wb') as f:
        pickle.dump(net, f)
    print('\nModel saved')
