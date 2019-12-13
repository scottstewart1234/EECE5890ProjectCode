import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import r2_score
import pandas as pd
import time

from extractGIST import BuildDataset

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def testModelsR(modelList, X, Y, varname="", folds=5, test_size_percentage=0.3):

    models_performance = []

    for mk, model in enumerate(modelList):
        print("Progress %s: Training model"%varname, mk+1, "out of", modelList.__len__())
        mtic = time.time()
        # iterate over folds
        scores = []

        for i in range(folds):
            print("\t* Model: ", mk+1, "Fold: ", i+1, "...", end="")
            sys.stdout.flush()
            ftic = time.time()

            # separate train and test set:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_percentage)

            # fitting model
            tmodel = model.fit(X_train, Y_train)

            # scores
            Y_predict = tmodel.predict(X_test)
            scores.append(r2_score(Y_test, Y_predict))

            ftoc = time.time() - ftic
            print(" done in %.3f [s]" % ftoc, end="\n")

        # accumulate scores
        scores = np.array(scores)
        models_performance.append(scores)

        mtoc = time.time() - mtic
        print("\t-- Model %d trained in: %.3f [s]" % (mk+1, mtoc))

    # data table formatting
    data = np.array(models_performance)

    return data

def plotBar(data, names, output="out", size=(10, 10)):

    mean = np.mean(data, axis=1)
    stdesv = np.std(data, axis=1)

    plt.figure(figsize=size)
    n_models = names.__len__()
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


def regress(X, LatLon, folds=5):

    linearModel = svm.SVR(kernel='linear')
    polinomial10 = svm.SVR(kernel='poly', degree=10, gamma='scale')
    polinomial50 = svm.SVR(kernel='poly', degree=50, gamma='scale')
    gaussian1 = svm.SVR(kernel='rbf', epsilon=0.22, gamma='scale')
    gaussian2 = svm.SVR(kernel='rbf', epsilon=0.54, gamma='scale')
    gaussian3 = svm.SVR(kernel='rbf', gamma='scale')
    sigmoid = svm.SVR(kernel='sigmoid', gamma=0.2)
    #modelList = [linearModel, polinomial10, gaussian1, gaussian2, gaussian3] #, sigmoid]
    #names = ("lineal", "poly10", "poly50", "gauss1", "gauss2", "gauss3", "sigmoid")

    modelList = [linearModel, gaussian1, gaussian2, gaussian3]
    names = ['linear', 'gauss1', 'gauss2', 'gauss3']

    dataLat = testModelsR(modelList, X, LatLon[:, 0], varname= "Latitude", folds=folds)
    dataLon = testModelsR(modelList, X, LatLon[:, 1], varname= "Longitude", folds=folds)

    dfLat = plotBar(dataLat, names, output='latitude', size=(10, 5))
    dfLon = plotBar(dataLon, names, output='longitude', size=(10, 5))

    print("Regression performance for Latitude:")
    print(dfLat)
    print("Regression performance for Longitude:")
    print(dfLon)

    return


if __name__ == "__main__":

    basefolder = os.path.join(".")
    json_path = os.path.join(basefolder, "index_roads.json")
    roads_path = os.path.join(basefolder, "roads")
    descriptors_folder = os.path.join(basefolder, "gist")
    X, Y, LatLon = BuildDataset(json_path, roads_path, descriptors_folder)

    folds = 30

    regress(X, LatLon, folds=folds)


