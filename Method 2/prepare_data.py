#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:42:23 2019

@author: scottstewart
"""
import os 
import numpy as np
import json
from keras.utils import np_utils
def one_hot_encode(y):

    # one hot encode outputs
    y = np_utils.to_categorical(y)
    num_classes = y.shape[1]
    return y,num_classes

def Encode(Y):
	Y2 = np.zeros((len(Y),2), dtype=np.float64)
	j=0
	for i in Y:
		lon = float(i[0])
		lat = float(i[1])
		lat = float(((lat)-26)/(49.3-25))
		lon = float(((lon)+66)/(-124.914+66))

		Y2[j,0]=lat
		Y2[j,1]=lon
		j+=1
	return Y2,2
directory = 'C:/Users/sstew/Desktop/Fall 2019/image-classification/dataset_image'
gistDirectory = 'C:/Users/sstew/Desktop/Fall 2019/image-classification/gist'
filenames = os.listdir(directory)
gist = []
y = []
f= []
l = []
with open('index_roads.json') as json_file:
    data = json.load(json_file)
for fil in filenames:
    image_label = str(fil)       
    worked = False
    image_label = str(image_label)
    try:
        lat = image_label[:-4].split('_')[0]
        lon = image_label[:-4].split('_')[1]
        
        for i in range(0,5622):
            
            if(image_label[:-3]==str(data[str(i)][2])[:-3]):
                j =str(format(i, '06d'))
                k=-1
                if(str(data[str(i)][3]) == 'Midwest'):
                    k = 0
                if(str(data[str(i)][3]) == 'Southeast'):
                    k = 1
                if(str(data[str(i)][3]) == 'West'):
                    k = 2
                if(str(data[str(i)][3]) == 'Southwest'):
                    k = 3
                if(str(data[str(i)][3]) == 'Northeast'):
                    k = 4
                 
                #print(j)
                addition = gistDirectory+'/'+j+'.npy'
                gist.append(addition)
                l.append([lat,lon])
                y.append(k)
                worked= True  
	    
    except:
        pass
    if(worked):
         f.append(fil)
print(len(gist))
y,_ = one_hot_encode(y)

y2 = []
for pointa,pointb in zip(y,l):
	y2.append([pointa[0],pointa[1],pointa[2],pointa[3],pointa[4],pointb[0],pointb[1]])
np.save('filenames.npy', f)
np.save('y_labels.npy', y2)
np.save('gist.npy',gist)
y2 = np.asarray(y2)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
filenames_shuffled, y_labels_shuffled,gist_shuffled = shuffle(f, y2,gist)

# saving the shuffled file.
# you can load them later using np.load().
np.save('y_labels_shuffled.npy', y_labels_shuffled)
np.save('filenames_shuffled.npy', filenames_shuffled)
np.save('gist_shuffled.npy',gist_shuffled)

filenames_shuffled_numpy = np.array(filenames_shuffled)
gist_shuffled_nump = np.array(gist_shuffled)

X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    filenames_shuffled_numpy, y_labels_shuffled, test_size=0.2, random_state=1)

gist_train_filenames, gist_val_filenames, y_train, y_val = train_test_split(
    gist_shuffled_nump, y_labels_shuffled, test_size=0.2, random_state=1)

print(X_train_filenames.shape) # (3800,)
print(y_train.shape)           # (3800, 12)


print(X_val_filenames.shape)   # (950,)
print(y_val.shape)             # (950, 12)

# You can save these files as well. As you will be using them later for training and validation of your model.
np.save('X_train_filenames.npy', X_train_filenames)
np.save('y_train.npy', y_train)

np.save('X_val_filenames.npy', X_val_filenames)
np.save('y_val.npy', y_val)

np.save('gist_train.npy',gist_train_filenames)
np.save('gist_val.npy',gist_val_filenames)

import shutil
shutil.make_archive("all_images", "zip", directory)
