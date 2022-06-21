import matplotlib.pyplot as plt
import numpy as np
import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
from skorch.helper import SliceDataset
from sklearn.model_selection import cross_val_score

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
    return random_split(dataset, [m - int(m / 4), int(m / 4)])


if __name__ == '__main__':
    print('COMP 472 Project')
    print('AI Face Mask Detector\n')
    # Hyper-parameters
    num_epochs = 2
    num_classes = 4
    learning_rate = 0.001
    classes = ('cloth', 'n95', 'none', 'surgical')

    train_dataset, _ = import_datasets()

    """Device to train"""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    """Start training"""
    y_train = np.array([np.int64(y) for x, y in iter(train_dataset)])

    torch.manual_seed(0)
    net = NeuralNetClassifier(
        CNN.CNN().to(device),
        max_epochs=num_epochs,
        lr=learning_rate,
        batch_size=64,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=device
    )
    net.fit(train_dataset, y=y_train)

    # """Predict"""
    # print('\n==== Predict ====')
    # y_predict = net.predict(test_dataset)
    # y_test = np.array([y for x, y in iter(test_dataset)])
    # print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_predict) * 100, 2)))
    # plot_confusion_matrix(net, test_dataset, y_test.reshape(-1, 1))
    # plt.show()

    # """K-fold cross-validate"""
    # print('\n==== K-fold ====')
    # train_sliceable = SliceDataset(train_dataset)
    # scores = cross_val_score(net, train_sliceable, y_train, cv=5, scoring='accuracy')
    # print('Scores: {}'.format(scores))

    """Save model state_dict"""
    net.save_params(f_params='model_pkl.pkl')
    print('\nModel state_dict saved')
