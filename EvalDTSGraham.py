'''
This script tests the full Graham model that won the CIFAR-10 challenge
Its features are:
    -center the color scale to the interval [-1,1]
    -padd zeros to increase the pictures to 126x126 pixels
'''

import numpy as np
import cPickle
import os
import keras
import json

import models

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

"""
Get data
"""

labels = [0,1,2,3,4,5,6,7,8,9]

score = []

for SIZE in range(50000,51000,5000):
    TRAININGSIZE    = SIZE
    TESTSIZE        = SIZE/10
    NUMBEROFLABELS  = len(labels)

    INPUTPATH = os.getcwd() + "/CIFAR-10/"

    x_train = []
    y_train = []

    for f in range(1,6):
        train = unpickle(INPUTPATH + "/data_batch_" + str(f))

        x = train["data"]
        y = train["labels"]

        for i in range(len(y)):
            if(len(y_train)==TRAININGSIZE):
                break
            if(y[i] in labels):
                x_train.append(x[i])
                y_train.append(y[i])

        if(len(y_train)==TRAININGSIZE):
            break

    x = np.array(x_train, float)
    y = np.array(y_train, int)

    x = np.reshape(x,(len(x),3,32,32))
    x = np.transpose(x,(0,3,1,2))
    x = np.transpose(x,(0,1,3,2))
    x = x*2/255. - 1

    x_train = np.zeros((len(x),128,128,3), float)
    x_train[:,48:80,48:80,:] = x

    y_train = keras.utils.to_categorical(y_train, NUMBEROFLABELS)

    test = unpickle(INPUTPATH + "/test_batch")

    x = test["data"]
    y = test["labels"]

    x_test = []
    y_test = []

    for i in range(len(y)):
        if(len(y_test) == TESTSIZE):
            break
        if(y[i] in labels):
            x_test.append(x[i])
            y_test.append(y[i])

    x = np.array(x_test, float)
    y = np.array(y_test, int)

    x = np.reshape(x,(len(x),3,32,32))
    x = np.transpose(x,(0,3,1,2))
    x = np.transpose(x,(0,1,3,2))
    x = x*2/255. - 1

    x_test = np.zeros((len(x),128,128,3),float)
    x_test[:,48:80,48:80,:] = x

    y_test = keras.utils.to_categorical(y_test, NUMBEROFLABELS)


    _ = models.Graham(x_train, y_train, x_test, y_test, NUMBEROFLABELS)
    # _ = models.Lenet(x_train, y_train, x_test, y_test)
    # _ = models.EERACN(x_train, y_train, x_test, y_test)
    print(_)

    score.append(_)

with open('score.dat','w') as outfile:
    for i in range(len(score)):
        outfile.write(str(score[i][0]) + "    " + str(score[i][1]) + "\n")
