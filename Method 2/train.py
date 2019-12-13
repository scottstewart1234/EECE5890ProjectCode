#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:52:08 2019

@author: scottstewart
"""

import numpy
import matplotlib.pyplot as plt
import keras
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Conv2D, Input, AveragePooling2D, UpSampling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
import load_data
from keras.models import Sequential, Model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from scipy import misc

K.set_image_dim_ordering('tf')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
def readDescriptor(descriptorPath):
    #"read descriptor from binary file"

    with open(descriptorPath, 'rb') as file:
        descriptor = pickle.load(file)
        print("features loaded  %d " % descriptor.__len__())

    return descriptor

def one_hot_encode(y):

    # one hot encode outputs
    y = np_utils.to_categorical(y)
    num_classes = y.shape[1]
    return y,num_classes

def pre_process(X):

    # normalize inputs from 0-255 to 0.0-1.0
    X=X.astype('float32')
    X = X / 255.0
    return X
def decode(lat, lon):
	lat = (float((((lat)/2)+0.5)*(49.3-25))+26)
	lon = (float((((lon)/2)+0.5)*(-124.914+66))-66)
	return lat, lon
def Encode(Y, encode=False):
	Y2 = numpy.zeros((len(Y),2), dtype=numpy.float64)
	j = 0
	for i in Y:
		
		lon = float(i[0])
		lat = float(i[1])
		if(encode):
			pass
			lat = (float(((lat)-26)/(49.3-25))-0.5)*2
			lon = (float(((lon)+66)/(-124.914+66))-0.5)*2

		Y2[j,0]=lat
		Y2[j,1]=lon
		j+=1
	return Y2,2
class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, gist, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.gist = gist
    
    
  def __len__(self) :
    return (numpy.ceil(len(self.image_filenames) / float(self.batch_size))).astype(numpy.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_gist = self.gist[idx * self.batch_size : (idx+1) * self.batch_size]
    A = numpy.array([
            (misc.imread('dataset_image/' + str(file_name)))
               for file_name in batch_x])
    #print(A[1].shape)
    b = []
    for element in A:
        C = element
        if(C.shape[0]>=1024 or C.shape[1]>1024):
            C = misc.imresize(C,(1024,1024))
        b.append( numpy.pad(C, ((0, 1024 -C.shape[0] ),(0, 1024- C.shape[1]),(0,0)), 'mean'))
    A = numpy.asarray(b)#, numpy.array([numpy.load(gist_name).reshape((960,1)) for gist_name in batch_gist])]
    return [A], numpy.array(batch_y)
               
def loadRange(start = 0, encode = True):
	# load data
	X,y=load_data.load_datasets(start)
	
	# pre process
	X=pre_process(X)
	
	#one hot encode
	y,num_classes=Encode(y, encode)
	#split datasetEncode
	return train_test_split(X, y, test_size=0.2, random_state=7)


def define_model(num_classes,epochs):
    # Create the model
    
    #added these
    
    inputsa = Input(shape=(1024, 1024, 3))
    #inputsb = Input(shape=(960,1))
    c1 = (Conv2D(32, (5, 5), padding='same', activation='linear', kernel_constraint=BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)))(inputsa)
    c2 =(Dropout(0.25))(c1)
    #c4 = keras.layers.subtract([c2,inputs])
    a1= (MaxPooling2D(pool_size=(2, 2)))(c2)
    
    c3 = (Conv2D(64, (3, 3), padding='same', activation='linear', kernel_constraint=BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)))(a1)
    c4 =(Dropout(0.25))(c3)
    #s1 = keras.layers.subtract([a1,c4])
    a2= (MaxPooling2D(pool_size=(2, 2)))(c4)
    
    c5 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))(a2)
    c6 = (Dropout(0.25))(c5)
    s1 = keras.layers.add([a2,c6])

    a3 = MaxPooling2D(pool_size=(2, 2))(s1)
    
    
    c7 =Conv2D(32, (3, 3), activation='sigmoid', padding='same', kernel_constraint=BatchNormalization(axis=-1,  epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))(a3)
    #c7 = keras.layers.subtract([c6,a3])
    a4 = MaxPooling2D(pool_size=(2, 2))(c7)
    
    
    #F1 = Flatten()(a1)
    F2 = Flatten()(a4)
    #F4 = Flatten()(inputsb)
    F3 = Flatten()(a3)
    u = keras.layers.concatenate([F2,F3])
    #u1 = keras.layers.concatenate([u,F2])
    #d1 = (Dense(1024,activation='sigmoid'))(u1)
    d2 = (Dense(128,activation='sigmoid'))(u)
    d3 = (Dense(64,activation='sigmoid'))(d2)
    out1 = Dense(32, activation='sigmoid')(d3)
    #d4 = (Dense(64,activation='sigmoid'))(u)
    #d5 = (Dense(32,activation='sigmoid'))(d4)
    #lon = Dense(5, activation='linear')(d5)
    out = Dense(5, activation='sigmoid')(out1)
    outputs= (out)
    # Compile model
    model = Model(inputs=[inputsa], outputs=outputs)
    lrate = 0.0000001
    decay = lrate/epochs
    sgd = SGD(lr=lrate, decay=1e-5, nesterov=False) #RMSprop(lr=lrate, rho=0.9, decay=1e-5, epsilon=None) 
    model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
    print(model.summary())
    return model





epochs = 25
#define model
model= define_model(2,epochs)


#X_train, X_test, y_train, y_test = loadRange(0)
# Fit the model
#for i in range(0,epochs):
#	j =0;
#	X_train, X_test, y_train, y_test = loadRange(j*1000)
#	print("EPOCH: "+str(i+1))
#	while(len(X_train)>10):
#		
#		history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1,)
#		j=j+1
#		X_train, X_test, y_train, y_test = loadRange(j*1000)
batch_size = 3
y_train = numpy.load("y_train.npy")[:,:5]
y_val = numpy.load("y_val.npy")[:,:5]
y_val_l = numpy.load("y_val.npy")[:,5:7]
#for y in y_train:
#	y[0] = y[0]-0.5
#	y[1] = y[1] -0.5
#for y in y_val:
#	y[0] = y[0]-0.5
#	y[1] = y[1] -0.5
	#y[0],y[1] = decode(y[0],y[1])
X_val_filenames = numpy.load("X_val_filenames.npy")
X_train_filenames = numpy.load("X_train_filenames.npy")
gist_train = numpy.load('gist_train.npy')
gist_val = numpy.load('gist_val.npy')

my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train,gist_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val,gist_val, batch_size)
model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(X_train_filenames.shape[0] // batch_size),
                   epochs = epochs,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(X_val_filenames.shape[0] // batch_size))

with open('error.csv', 'a+', newline='') as thefile:
	j=0;
	X_train, X_test, y_train, y_test = loadRange(j*1000, encode = False)
	y_pred = model.predict_generator(my_validation_batch_generator)
	
	#while(len(X_train)>10):
	for i in range(0,len(y_val)):
			#lat, lon = #y_pred[i,0], y_pred[i,1]#decode(y_pred[i,0],y_pred[i,1])
			line  = ""+str(y_val[i,0]) +","+str(y_val[i,1]) +","+str(y_val[i,2]) +","+str(y_val[i,3]) +","+str(y_val[i,4])+','+str(y_val_l[i,0])+','+str(y_val_l[i,1])
			line2 = ""+str(y_pred[i,0])+","+str(y_pred[i,1])+","+str(y_pred[i,2])+","+str(y_pred[i,3])+","+str(y_pred[i,4]);
			line = line+","+line2+'\n'
			thefile.write(line)
		#j=j+1
		#X_train, X_test, y_train, y_test = loadRange(j*1000)
	thefile.close()



# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)


# serialize model to JSONx
model_json = model.to_json()
with open("model_face.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_face.h5")
print("Saved model to disk")

from keras.utils import plot_model
plot_model(model, to_file='model.png')
