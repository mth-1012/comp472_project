# comp472_project

AI Face Mask Detector

Members:
- Thanh Ta (40085781)
- Madline Nessim (40078034)
- Lam Tran (40088195)
- Hung Cai (40123967)


There are 3 main files are main.py, CNN.py and evaluator.py and a folder called "old".

The folder "old" contains the old versions of main.py and evaluator.py. 

main.py is where we load the dataset, transform it and then feed it to our model to train it. 
CNN.py is the convolutional neural network model that studies and analyzes the dataset
evaluator.py is the model evaluator, it runs over the test dataset and evaluates the model predictions using k-fold. 
It also calculates the accuracy, precision, recall, and F1-measure and outputs the confusion matrix for each class.

To run it the main.py usually has to run first then the evaluator.py but, since the main.py has been run before and
the model is already built (model.pt) and restored, only the evaluator.py has to run on the testing dataset.

If this error 
OMP: Error #15: Initializing libiomp5md.dll, but found mk2iomp5md.dll already initialized.
shows when running the evaluator.py, you will need to add this:

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
