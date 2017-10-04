import cPickle
import random
import os
import numpy as np
import keras

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def loadCIFAR10(size,labels,freq=0,val_split=0.2):
    """
    Input:
        SIZE:       size of the training data, including the validation set
        labels:     which labels to use
        freq:       relative frequency of the classes, if let to zero every class will
                    appear equally often
        val_split:  how much of the training data should be used for validation
    Output:
        (x_train,y_train,x_val,y_val,x_test,y_test)
    """

    if(freq==0):
        freq = [1./len(labels)] * len(labels)

    fsum    = sum(freq)
    NOL     = len(labels)

    INPUTPATH = os.getcwd() + "/../CIFAR-10"

    x_train = []
    y_train = []

    ##
    ## Load data into x_train and y_train
    ##

    for f in range(1,6):
        train = unpickle(INPUTPATH + "/data_batch_" + str(f))

        x = train["data"]
        y = train["labels"]

        for i in range(len(y)):
            if(len(y_train)==size*fsum):
                break
            if(y[i] in labels):
                if(y_train.count(y[i])<=size*freq[labels.index(y[i])]):
                    x_train.append(x[i])
                    y_train.append(y[i])

        if(len(y_train)==size*fsum):
            break

    ##
    ## Bring data into the correct format
    ##

    x_train = np.array(x_train, float)
    y_train = np.array(y_train, int)

    x_train = np.reshape(x_train,(len(x_train),3,32,32))
    x_train = np.transpose(x_train,(0,3,1,2))
    x_train = np.transpose(x_train,(0,1,3,2))

    x_train = x_train*2./255. - 1.


    y_train = keras.utils.to_categorical(y_train, NOL)

    ##
    ## shuffle data in case the original data was in order, preventing
    ## filling the validation set with only one class
    ##

    tmp = zip(x_train,y_train)
    random.shuffle(tmp)

    x_train,y_train = zip(*tmp)

    ##
    ## Split data into training set and validation set
    ##

    x_val = x_train[int(size*(1-val_split)):-1]
    y_val = y_train[int(size*(1-val_split)):-1]
    x_train = x_train[0:int(size*(1-val_split))]
    y_train = y_train[0:int(size*(1-val_split))]

    test = unpickle(INPUTPATH + "/test_batch")

    x = test["data"]
    y = test["labels"]

    x_test = []
    y_test = []

    for i in range(len(y)):
        if(len(y_test) == size*fsum/10.):
            break
        if(y[i] in labels):
            if(y_test.count(y[i])<=size*freq[labels.index(y[i])]/10.):
                x_test.append(x[i])
                y_test.append(y[i])

    x_test = np.array(x_test, float)
    y_test = np.array(y_test, int)

    x_test = np.reshape(x_test,(len(x_test),3,32,32))
    x_test = np.transpose(x_test,(0,3,1,2))
    x_test = np.transpose(x_test,(0,1,3,2))

    x_test = x_test*2./255. - 1.

    y_test = keras.utils.to_categorical(y_test, NOL)

    return (x_train,y_train,x_val,y_val,x_test,y_test)
