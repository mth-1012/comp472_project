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
from evaluator import predict_eval

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(32),
    transforms.CenterCrop(32),
])
classes = ('cloth', 'n95', 'none', 'surgical')


def import_datasets():
    """Import datasets"""
    dataset = torchvision.datasets.ImageFolder(root='./data/dataset/', transform=transform)

    """Random split data"""
    m = len(dataset)
    train, test = random_split(dataset, [m - int(m / 4), int(m / 4)])
    return train, test


if __name__ == '__main__':
    print('COMP 472 Project')
    print('AI Face Mask Detector\n')

    # Hyper-parameters
    num_epochs = 10
    batch_size = 100
    learning_rate = 1e-3

    """Import training dataset"""
    train_dataset, test_dataset = import_datasets()

    """Check device to train"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    """Start training"""
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        CNN.CNN(),
        max_epochs=num_epochs,
        lr=learning_rate,
        batch_size=batch_size,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=device,
        iterator_train__shuffle=True,
    )
    y_train = np.array([np.int64(y) for x, y in iter(train_dataset)])
    net.fit(train_dataset, y=y_train)

    """Save model state_dict"""
    with open('model.pkl', 'wb') as f:
        pickle.dump(net, f)
    print('\nModel saved')

    """Evaluate test dataset"""
    predict_eval(net, test_dataset, 'Test (25%)')
