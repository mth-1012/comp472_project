import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from skorch.helper import SliceDataset
from sklearn.model_selection import cross_val_score
from main import import_datasets, transform


def predict_eval(net, dataset):
    print('\n==== Predict ====')
    y_predict = net.predict(dataset)
    y_test = np.array([y for x, y in iter(test_dataset)])
    print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_predict) * 100, 2)))
    plot_confusion_matrix(net, test_dataset, y_test.reshape(-1, 1))
    plt.show()


def k_fold_cross_validation(net, dataset, k=5):
    print('\n==== K-fold ====')
    y_train = np.array([np.int64(y) for x, y in iter(dataset)])
    train_sliceable = SliceDataset(dataset)
    scores = cross_val_score(net, train_sliceable, y_train, cv=k, scoring='accuracy')
    print('Scores: {}'.format(scores))


if __name__ == '__main__':
    print('COMP 472 Project')
    print('Evaluator')

    """Check device to evaluate"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    """Get datasets"""
    data, _, test_dataset = import_datasets()

    """Import bias datasets"""
    male_dataset = torchvision.datasets.ImageFolder(root='./bias/gender/male/', transform=transform)
    female_dataset = torchvision.datasets.ImageFolder(root='./bias/gender/female/', transform=transform)
    child_dataset = torchvision.datasets.ImageFolder(root='./bias/age/child/', transform=transform)
    adult_dataset = torchvision.datasets.ImageFolder(root='./bias/age/adult/', transform=transform)
    senior_dataset = torchvision.datasets.ImageFolder(root='./bias/age/senior', transform=transform)

    """Reload model"""
    with open('model-pkl.pkl', 'rb') as f:
        net_reload = pickle.load(f)
        print('\nModel loaded')

    """Evaluate performance"""
    predict_eval(net_reload, test_dataset)

    """K-fold Cross-Validate"""
    k_fold_cross_validation(net_reload, data, k=5)

