import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
import time

from extractGIST import BuildDataset

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def testModelsC(modelList, X, Y, folds=5, test_size_percentage=0.3):

    models_performance = []
    models_confusion =[]
    for mk, model in enumerate(modelList):
        print("Progress: Training model", mk+1, "out of", modelList.__len__())
        mtic = time.time()
        # iterate over folds
        scores = []
        cm = []
        for i in range(folds):
            print("\t* Model: ", mk+1, "Fold: ", i+1, "...", end="")
            sys.stdout.flush()
            ftic = time.time()

            # separate train and test set:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_percentage)
            # fitting model
            tmodel = model.fit(X_train, Y_train)
            scores.append(model.score(X_test, Y_test))
            # compute confusion matrix
            Y_predict = tmodel.predict(X_test)
            #cm.append(confusion_matrix(Y_test, Y_predict))

            # save y_test and y_predict
            tablecf = {"Y_Test": Y_test, "Y_Pred": Y_predict}
            dfm = pd.DataFrame(tablecf)
            dfm.to_csv('csvs/m%d'%(mk+1) + '_f%d'%(i+1) + '.csv')

            ftoc = time.time() - ftic
            print(" done in %.3f [s]" % ftoc, end="\n")

        # stack confusion matrices
        #confm = np.stack(cm, axis=0)
        #confmean = confm.mean(axis=0)

        # accumulate scores
        scores = np.array(scores)
        models_performance.append(scores)
        #models_confusion.append(confmean)
        mtoc = time.time() - mtic
        print("Progress: Model %d trained in: %.3f [s]" % (mk+1, mtoc))

    # data table formatting
    data = np.array(models_performance)
    conf = 0 #np.array(models_confusion)

    return data , conf

def plotBar(data, n_models, output="out", size=(10, 10)):

    mean = np.mean(data, axis=1)
    stdesv = np.std(data, axis=1)

    plt.figure(figsize=size)
    names = ("lineal", "poly10", "poly50", "gauss1", "gauss2", "gauss3", "sigmoid")
    x_aux = np.linspace(0, n_models - 1, n_models, dtype=np.int32)
    plt.bar(x_aux, mean)
    plt.errorbar(x_aux, mean, fmt='o', yerr=stdesv, uplims=True, lolims=True, label="average model performance",
                 ecolor="red", elinewidth=3.0, barsabove=True)
    plt.xticks(np.arange(n_models), names[:n_models], fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Tested models", fontsize=20)
    plt.ylabel("Performance [%]", fontsize=20)
    plt.grid(which="both", linestyle='-')
    plt.savefig(output+'.svg', format='svg')

    table = {"Model name": names[:n_models],
             "Mean performance": mean,
             "Standard dev": stdesv
             }
    df = pd.DataFrame(table, columns=["Model name", "Mean performance", "Standard dev"])
    df.to_csv(output+'.csv')

    return df

def classify(X, Y, folds=5):

    linearModel = svm.SVC(kernel='linear')
    polinomial10 = svm.SVC(kernel='poly', degree=10)
    polinomial50 = svm.SVC(kernel='poly', degree=50)
    gaussian1 = svm.SVC(kernel='rbf', gamma=0.22)
    gaussian2 = svm.SVC(kernel='rbf', gamma=0.54)
    gaussian3 = svm.SVC(kernel='rbf', gamma='scale')
    sigmoid = svm.SVC(kernel='sigmoid', gamma=2)
    modelList = [linearModel, polinomial10, polinomial50, gaussian1, gaussian2, gaussian3, sigmoid]

    dataClass, confC = testModelsC(modelList, X, Y, folds=folds, test_size_percentage=0.3)
    dfClass = plotBar(dataClass, modelList.__len__(), output='classifier', size=(10, 10))

    print("Classification Performance Table:")
    print(dfClass)

    return

if __name__ == "__main__":

    basefolder = os.path.join(".")
    json_path = os.path.join(basefolder, "index_roads.json")
    roads_path = os.path.join(basefolder, "roads")
    descriptors_folder = os.path.join(basefolder, "gist")
    X, Y, LatLon = BuildDataset(json_path, roads_path, descriptors_folder)

    folds = 30

    classify(X, Y, folds=folds)



