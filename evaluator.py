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
from main import import_datasets


def predict_eval(net, dataset):
    print('\n==== Predict ====')
    y_predict = net.predict(dataset)
    y_test = np.array([y for x, y in iter(test_dataset)])
    print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_predict) * 100, 2)))
    plot_confusion_matrix(net, test_dataset, y_test.reshape(-1, 1))
    plt.show()


if __name__ == '__main__':
    print('COMP 472 Project')
    print('Evaluator')

    """Device to evaluate"""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    """Get random test dataset"""
    _, test_dataset = import_datasets()

    """Reload model"""
    net_reload = NeuralNetClassifier(
        CNN.CNN(),
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=device
    )
    net_reload.initialize()
    net_reload.load_params(f_params='model_pkl.pkl')

    """Predict"""
    predict_eval(net_reload, test_dataset)
