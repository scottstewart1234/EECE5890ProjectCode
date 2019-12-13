"""
Module to retrieve GIST features of an image.
"""

import gist
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt


def JSON2dict(JSONpath):
    try:
        with open(JSONpath) as json_file:
            infoDict = json.load(json_file)
    except:
        raise Exception(".json file not found at %s" % JSONpath)

    return infoDict


def extractDescriptor(imgID, imgFilePath, outputFolder, forceUpdate=False):
    """Extract requested descriptor and store in binary file
    :param imgID        -> ID of the selected image
    :param imgFilePath  -> dictionary with information about lat, lon, name and region
    :param outputFolder -> Output for binary descriptor file
    :param forceUpdate  -> force file update
     """

    # 1 descriptor file name
    descriptorFileName = os.path.join(outputFolder, "%06d" % int(imgID) + ".npy")

    # 1.5 If exist and forceupdate, delete it.
    if os.path.isfile(descriptorFileName) and forceUpdate:
        os.remove(descriptorFileName)

    # 2 compute it if it doesn't exist
    if not (os.path.isfile(descriptorFileName) and not forceUpdate):
        descriptor = gist.extract(plt.imread(imgFilePath))
        try:
            file = open(descriptorFileName, "wb")  # file creation
            pickle.dump(descriptor, file)
            file.close()
            print("features created %d " % descriptor.__len__())
        except:
            raise Exception("error_writing_%s" % outputFolder + descriptorFileName)
    else:
        descriptor = readDescriptor(descriptorFileName)
    return descriptor


def readDescriptor(descriptorPath):
    "read descriptor from binary file"

    with open(descriptorPath, 'rb') as file:
        descriptor = pickle.load(file)
        print("features loaded  %d " % descriptor.__len__())

    return descriptor


def BuildDataset(jsonpath, roadsFolder,  descriptorsFolder, featureSize=960):
    region_name_dict = {"Northeast": 1, "Midwest": 2, "Southeast": 3, "Southwest": 4, "West": 5}
    infoDict = JSON2dict(jsonpath)
    read_size = len(infoDict)
    X = np.zeros([read_size, featureSize])
    Y = np.zeros([read_size])
    LatLon = np.zeros([read_size, 2])
    counter = 0
    failedFilesList = []
    for keyID in infoDict:
        try:
            # 1. Get descriptor
            imgFileName = infoDict[keyID][2]
            imgFilePath = os.path.join(roadsFolder, imgFileName)
            X[counter] = extractDescriptor(keyID, imgFilePath, descriptorsFolder)
            # 2. Read region label
            Y[counter] = region_name_dict[infoDict[keyID][3]]
            LatLon[counter] = np.array([infoDict[keyID][0], infoDict[keyID][1]])
            counter += 1
            print("example: %s" % counter)
            if counter >= read_size:
                break
        except KeyError:
            print("Key error: %s" % keyID)
            failedFilesList.append(keyID)
        except:
            print("Other error for key: %s" % keyID)
            failedFilesList.append(keyID)

    print("Failed keys:", failedFilesList)
    return X, Y, LatLon


if __name__ == "__main__":

    # Determine paths
    basefolder = os.path.join("./")
    roads_path = os.path.join(basefolder, "roads")
    descriptors_dir = os.path.join(basefolder, "gist")
    # json index file
    infoDict = JSON2dict(os.path.join(basefolder, "index_roads.json"))

    # All descriptors
    for keyID in infoDict:

    # some descriptors
    #for keyID in ['488', '842', '1925', '2303', '2514', '2606', '4010', '4102', '5251']:

        print("%06d" % int(keyID), end=": ")
        imgFileName = infoDict[keyID][2]
        imgFilePath = os.path.join(roads_path, imgFileName)
        extractDescriptor(keyID, imgFilePath, descriptors_dir, forceUpdate=True)


