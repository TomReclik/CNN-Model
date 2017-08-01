import numpy as np
import cPickle
import os
import keras

import models

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

INPUTPATH = os.getcwd() + "/CIFAR-10/"
train = unpickle(INPUTPATH + "train")

x_train = train['data']
y_train = np.array(train['labels'])

x_train = np.reshape(x_train,(len(x_train),3,32,32))
x_train = np.transpose(x_train,(0,3,1,2))
x_train = np.transpose(x_train,(0,1,3,2))
x_train = x_train/255.

y_train = keras.utils.to_categorical(y_train, 5)


test = unpickle(INPUTPATH + "test")

x_test = test['data']
y_test = np.array(test['labels'])

x_test = np.reshape(x_test,(len(x_test),3,32,32))
x_test = np.transpose(x_test,(0,3,1,2))
x_test = np.transpose(x_test,(0,1,3,2))
x_test = x_test/255.

y_test = keras.utils.to_categorical(y_test, 5)

score = models.SCFN(x_train, y_train, x_test, y_test)

print score
