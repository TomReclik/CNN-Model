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

labels = [0,1,2,3,4]

score = []

for SIZE in range(5000,26000,5000):
    TRAININGSIZE    = SIZE
    TESTSIZE        = SIZE/10


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

    x_train = np.array(x_train, float)
    y_train = np.array(y_train, int)

    x_train = np.reshape(x_train,(len(x_train),3,32,32))
    x_train = np.transpose(x_train,(0,3,1,2))
    x_train = np.transpose(x_train,(0,1,3,2))
    x_train = x_train/255.

    y_train = keras.utils.to_categorical(y_train, 5)

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

    x_test = np.array(x_test, float)
    y_test = np.array(y_test, int)

    x_test = np.reshape(x_test,(len(x_test),3,32,32))
    x_test = np.transpose(x_test,(0,3,1,2))
    x_test = np.transpose(x_test,(0,1,3,2))
    x_test = x_test/255.

    y_test = keras.utils.to_categorical(y_test, 5)


    _ = models.Graham(x_train, y_train, x_test, y_test)
    # _ = models.Lenet(x_train, y_train, x_test, y_test)
    # _ = models.EERACN(x_train, y_train, x_test, y_test)
    print(_)

    score.append(_)

with open('score.dat','w') as outfile:
    for i in range(len(score)):
        outfile.write(str(score[i][0]) + "    " + str(score[i][1]) + "\n")
