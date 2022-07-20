import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from skorch.helper import SliceDataset
from sklearn.model_selection import cross_validate
import PySimpleGUI as sg
import os.path
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(32),
    transforms.CenterCrop(32),
])
classes = ('cloth', 'n95', 'none', 'surgical')


def predict_eval(net, dataset, name):
    print('\n==== Predict ====')
    y_predict = net.predict(dataset)
    y_test = np.array([y for x, y in iter(dataset)])
    print('Accuracy for {} Dataset: {}%'.format(name, round(accuracy_score(y_test, y_predict) * 100, 2)))
    ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=classes)
    plt.title('Confusion Matrix for {} Dataset'.format(name))
    plt.show()


def k_fold_cross_validation(net, dataset, k=5):
    print('\n==== K-fold ====')
    print('Warning: This takes approximately an hour or more!')
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='weighted'),
    }
    y_train = np.array([np.int64(y) for x, y in iter(dataset)])
    train_sliceable = SliceDataset(dataset)
    scores = cross_validate(net, train_sliceable, y_train, cv=k, scoring=scoring)
    print('\nScores:')
    print('Accuracy: ', format(scores['test_accuracy']))
    print('Precision: ', format(scores['test_precision']))
    print('Recall: ', format(scores['test_recall']))
    print('F1-measure: ', format(scores['test_f1_score']))

if __name__ == '__main__':
    print('COMP 472 Project')
    print('Extra Evaluator')
    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
    ]

    image_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                   and f.lower().endswith((".png", ".gif", ".jpg"))
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                #Path to the chosen file
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                window["-TOUT-"].update(filename)
                window["-IMAGE-"].update(filename=filename)
                window.close()
            except:
                pass
    """Check device to evaluate"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    """Get datasets"""
    data = torchvision.datasets.ImageFolder(root='./data/dataset/', transform=transform)

    """Import bias datasets"""
    # male_dataset = torchvision.datasets.ImageFolder(root='./data/bias/gender/male/', transform=transform)
    # female_dataset = torchvision.datasets.ImageFolder(root='./data/bias/gender/female/', transform=transform)
    # child_dataset = torchvision.datasets.ImageFolder(root='./data/bias/age/child/', transform=transform)
    # adult_dataset = torchvision.datasets.ImageFolder(root='./data/bias/age/adult/', transform=transform)
    # senior_dataset = torchvision.datasets.ImageFolder(root='./data/bias/age/senior', transform=transform)

    """Reload model"""
    with open('model.pkl', 'rb') as f:
        net_reload = pickle.load(f)
        print('\nModel loaded')

    # """Evaluate performance"""
    # # predict_eval(net_reload, male_dataset, 'Male (Gender)')
    # # predict_eval(net_reload, female_dataset, 'Female (Gender)')
    # # predict_eval(net_reload, child_dataset, 'Child (Age)')
    # # predict_eval(net_reload, adult_dataset, 'Adult (Age)')
    # # predict_eval(net_reload, senior_dataset, 'Senior (Age)')
    #
    # """K-fold Cross-Validate"""
    # k_fold_cross_validation(net_reload, data, k=10)
