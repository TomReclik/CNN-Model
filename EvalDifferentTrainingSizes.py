import numpy as np
import cPickle
import os
import keras
import json
import tensorflow as tf

import models
import DifferentStructures

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

flags = tf.app.flags
flags.DEFINE_string('CenterGrayScale', 'True', 'Center the gray scale to the interval -1,1. Bool.')

FLAGS = flags.FLAGS

labels = [0,1,2,3,4,5]

#
# All categarozies equally often
#
dist = [1./len(labels)] * len(labels)
# dist = [0.0799,0.1076,0.4132,0.0035,0.3229,0.0729]

DSUM = sum(dist)

score = []

#
# Low number of examples range
#

# low = range(500,5100,500)

#
# High number of examples range
#
#
# high = range(5000,26000,5000)
#
# RANGE = low+high

RANGE = [5000]

#
# Number of repetitions for statistical evaluation
#

NOR = 20

for SIZE in RANGE:
    #
    # Number of elements per class
    #
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
            if(len(y_train)==TRAININGSIZE*DSUM):
                break
            if(y[i] in labels):
                if(y_train.count(y[i])<=TRAININGSIZE*dist[labels.index(y[i])]):
                    x_train.append(x[i])
                    y_train.append(y[i])

        if(len(y_train)==TRAININGSIZE*DSUM):
            break

    x_train = np.array(x_train, float)
    y_train = np.array(y_train, int)

    x_train = np.reshape(x_train,(len(x_train),3,32,32))
    x_train = np.transpose(x_train,(0,3,1,2))
    x_train = np.transpose(x_train,(0,1,3,2))
    if(FLAGS.CenterGrayScale):
        x_train = x_train*2./255. - 1.
    else:
        x_train = x_train/255.

    y_train = keras.utils.to_categorical(y_train, NUMBEROFLABELS)

    test = unpickle(INPUTPATH + "/test_batch")

    x = test["data"]
    y = test["labels"]

    x_test = []
    y_test = []

    for i in range(len(y)):
        if(len(y_test) == TESTSIZE*DSUM):
            break
        if(y[i] in labels):
            if(y_test.count(y[i])<=TESTSIZE*dist[labels.index(y[i])]):
                x_test.append(x[i])
                y_test.append(y[i])

    x_test = np.array(x_test, float)
    y_test = np.array(y_test, int)

    x_test = np.reshape(x_test,(len(x_test),3,32,32))
    x_test = np.transpose(x_test,(0,3,1,2))
    x_test = np.transpose(x_test,(0,1,3,2))
    if(FLAGS.CenterGrayScale):
        x_test = x_test*2./255. - 1.
    else:
        x_test = x_test/255.

    y_test = keras.utils.to_categorical(y_test, NUMBEROFLABELS)

    # _ = models.Graham_Simple(x_train, y_train, x_test, y_test, NUMBEROFLABELS)
    # _ = models.Lenet(x_train, y_train, x_test, y_test)
    # _ = models.CCIFC(x_train, y_train, x_test, y_test, NUMBEROFLABELS)

    _ = []
    for i in range(NOR):
        _.append(models.Graham_Simple(x_train, y_train, x_test, y_test, NUMBEROFLABELS))

    print _[0]
    score.append(_)

with open('Graham_Simple_stat.dat','w') as outfile:
    for i in range(len(score)):
        outfile.write("Number of training data: " + str(RANGE[i]) + "\n")
        outfile.write("Loss: ")
        for j in range(len(score[i])):
            outfile.write("%f" %score[i][j][0])
            if(j<len(score[i])-1):
                outfile.write(", ")
        outfile.write("\n")

        outfile.write("Accuracy: ")
        for j in range(len(score[i])):
            outfile.write("%f" %score[i][j][1])
            if(j<len(score[i])-1):
                outfile.write(", ")
        outfile.write("\n")
