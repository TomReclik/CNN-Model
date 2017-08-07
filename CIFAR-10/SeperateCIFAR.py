import numpy as np
import cPickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

#
# TRAININGSIZE  : Number of training samples
# TESTSIZE      : Number of test samples
# INPUTPATH     : Path to data
#

TRAININGSIZE    = 50000
TESTSIZE        = 5000
INPUTPATH       = os.getcwd()

labels = [0,1,2,3,4]

#
# Seperate training data
#

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

train = {"data": x_train, "labels": y_train}

with open('train','w') as outfile:
    cPickle.dump(train, outfile)

#
# Seperate test data
#

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

test = {"data": x_test, "labels": y_test}

with open('test','w') as outfile:
    cPickle.dump(test, outfile)

print("Training size: ", len(x_train))
print("Test size:", len(x_test))
