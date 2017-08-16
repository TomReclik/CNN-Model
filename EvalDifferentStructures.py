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


"""
Get data
"""

labels = [0,1,2,3,4]

SIZE = 1000

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

x_train = np.array(x_train, float)
y_train = np.array(y_train, int)

x_train = np.reshape(x_train,(len(x_train),3,32,32))
x_train = np.transpose(x_train,(0,3,1,2))
x_train = np.transpose(x_train,(0,1,3,2))
if(FLAGS.CenterGrayScale):
    print("Centering data")
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
if(FLAGS.CenterGrayScale):
    x_test = x_test*2./255. - 1.
else:
    x_test = x_test/255.

y_test = keras.utils.to_categorical(y_test, NUMBEROFLABELS)

# scoreMPIC = DifferentStructures.MaxPoolIncreasingChannel(x_train, y_train, x_test, y_test, NUMBEROFLABELS, 180)
#
# print("MaxPoolIncreasingChannel")
# print(scoreMPIC)
#
# with open('score_MaxPoolIncreasingChannel.dat','w') as outfile:
#     outfile.write(str(scoreMPIC[0]) + "    " + str(scoreMPIC[1]) + "\n")

# scoreMPCC = DifferentStructures.MaxPoolConstantChannel(x_train, y_train, x_test, y_test, NUMBEROFLABELS, 180)
#
# print("MaxPoolConstantChannel")
# print(scoreMPCC)
#
# with open('score_MaxPoolConstantChannel.dat','w') as outfile:
    # outfile.write(str(scoreMPCC[0]) + "    " + str(scoreMPCC[1]) + "\n")

_ = models.EERACN(x_train, y_train, x_test, y_test, NUMBEROFLABELS)
print(_)

# scoreNMPCC = DifferentStructures.NoMaxPoolConstantChannel(x_train, y_train, x_test, y_test, NUMBEROFLABELS, 180)
#
# print("NoMaxPoolConstantChannel")
# print(scoreNMPCC)
#
# with open('score_NoMaxPoolConstantChannel.dat','w') as outfile:
#     outfile.write(str(scoreNMPCC[0]) + "    " + str(scoreNMPCC[1]) + "\n")
