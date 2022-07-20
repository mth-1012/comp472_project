import pickle
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import PySimpleGUI as sg
import os.path
import tensorflow as tf

def model_evaluate_single(model, image_dir, transform, classes):
    print('==== single evaluate ====')
    image = tf.keras.preprocessing.image.load_img(image_dir)
    image = transform(image)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    print(predictions)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(32),
    transforms.CenterCrop(32),
])
classes = ('cloth', 'n95', 'none', 'surgical')

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
                   and f.lower().endswith((".png", ".gif", ".jpg", ".webp"))
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                #Path to the chosen file
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                window["-TOUT-"].update(filename)
                window.close()
            except:
                pass

    print(filename)
    """Get datasets"""
    data = torchvision.datasets.ImageFolder(root='./data/dataset/', transform=transform)

    """Reload model"""
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        print('\nModel loaded')

    model_evaluate_single(model, filename, transform, classes)





