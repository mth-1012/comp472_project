# AI Face Mask Detection

Part of the COMP 472 course at Concordia University

| Member | GitHub |
|----------|--------------------|
| Thanh Ta | [mth-1012](https://github.com/mth-1012) |
| Madline Nessim | [madeleine341](https://github.com/madeleine3341) |
| Lam Tran | [linchen2508](https://github.com/linchen2508) |
| Hung Cai | [Pro-vo-ker](https://github.com/Pro-vo-ker) |

## Files

**main.py** is where we load the dataset, transform it, and then feed it to our model to train

**CNN.py** is the convolutional neural network model that studies and analyzes the dataset

**evaluator.py** is where the evaluator functions are stored, including train/test split, k-fold cross-validation and
single evaluation (to be added); bias evaluation is in the code but commented out

## Libraries

Using Anaconda, install the following before running (note that _cudatoolkit_ is not 
mandatory and can be omitted, it is around 625 MB)

    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

Continue to install the following tools.

    pip install skorch

Make sure _jupyter notebook_ is installed before running the notebook file.

## Previous version

The folder "old" contains the old version of main and evaluator, where we didn't utilize _**scikit-learn**_

## How to run

There are two ways to test run, pure Python or Jupyter Notebook environment

### Pure Python

Run the main.py with the IDE or in the console, enter 

    python main.py

After the training finishes, a train/test split (75-25) evaluation is automatically called and the result will be shown. 
It should be similar to this

    Model saved
    
    ==== Predict ====
    Accuracy for Test (25%) Dataset: 80.55%

For further evaluation, run the evaluator.py with the IDE or in the console, enter

    python evaluator.py

### Jupyter Notebook

(to be updated)

## Known Error

If the error 

    OMP: Error #15: Initializing libiomp5md.dll, but found mk2iomp5md.dll already initialized.

is shown when running the evaluator.py, add

    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

to the appropriate position in main.py.