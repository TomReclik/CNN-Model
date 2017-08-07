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

"""
Different combinations of parameters
"""
#
# channels = [[16,16,16],
#             [16,16,32],
#             [16,32,32],
#             [32,32,32],
#             [32,32,64],
#             [32,64,64],
#             [64,64,64]]
#
# dropout = [[0.2,0.3,0.4,0.5]]
#
# dense   = [64,128,256]
#
# SCORES = dict()
#
# for i in range(len(channels)):
#     for j in range(len(dropout)):
#         for k in range(len(dense)):
#             score = models.SCFN(x_train, y_train, x_test, y_test,
#                                 channels[i], dropout[j], dense[k])
#             tmpdict = { "Channels": channels[i],
#                         "Dropout" : dropout[j],
#                         "Dense"   : dense[k],
#                         "Score"   : score
#                         }
#             print("SCORE: ")
#             print(score)
#             name    = str(i) + str(j) + str(k)
#             SCORES[name] = tmpdict
#
#
# with open('scores.txt','w') as f:
#     json.dump(SCORES, f)

# channels = [[16,16],
#             [16,32],
#             [16,64],
#             [32,16],
#             [32,32],
#             [32,64],
#             [64,16],
#             [64,32],
#             [64,64]]

# channels = [[32,64]]
#
# for i in range(len(channels)):
#     score = models.TCFN(x_train, y_train, x_test, y_test, channels[i])
#     print score

score = models.EERACN(x_train, y_train, x_test, y_test)
print score
