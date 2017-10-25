import cPickle
import random
import os
import numpy as np
import keras
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

class Siamese_Loader:
    """
    Loads the data and returns batches of given size
    """
    def __init__(self,size,labels,path):
        """
        Input:
            size:       size of the training data, including the validation set
            labels:     which labels to use
            path:       path to data
        """

        self.data = {}

        train_len = [size/len(labels)] * len(labels)

        # if(freq==0):
        #     train_len = [size/len(labels)] * len(labels)

        INPUTPATH = os.getcwd() + "/../CIFAR-10"

        x_tmp = []
        y_tmp = []

        ##
        ## Load data into x_train and y_train
        ##

        for f in range(1,6):
            train = unpickle(INPUTPATH + "/data_batch_" + str(f))

            x = train["data"]
            y = train["labels"]

            for i in range(len(y)):
                if(len(y_tmp)==size):
                    break
                if(y[i] in labels):
                    if(y_tmp.count(y[i])<train_len[y[i]]):
                        x_tmp.append(x[i])
                        y_tmp.append(y[i])

            if(len(y_tmp)==size):
                break

        ##
        ## Bring data into the correct format
        ##

        x_tmp = np.array(x_tmp, float)
        y_tmp = np.array(y_tmp, int)

        x_tmp = np.reshape(x_tmp,(len(x_tmp),3,32,32))
        x_tmp = np.transpose(x_tmp,(0,2,3,1))

        # x_tmp = x_tmp*2./255. - 1.

        ##
        ## Transform data into shape for siamese networks
        ##

        x_train = np.zeros((len(labels),int(size/len(labels)),32,32,3), dtype=float)

        pos = np.zeros(len(labels), dtype=int)

        for i in range(len(x_tmp)):
            x_train[y_tmp[i]][pos[y_tmp[i]]] = x_tmp[i]
            pos[y_tmp[i]] += 1

        self.data["train"] = x_train

        test = unpickle(INPUTPATH + "/test_batch")

        x = test["data"]
        y = test["labels"]

        x_tmp = []
        y_tmp = []

        for i in range(len(y)):
            if(len(y_tmp) == size/10):
                break
            if(y[i] in labels):
                if(y_tmp.count(y[i])<train_len[y[i]]/10):
                    x_tmp.append(x[i])
                    y_tmp.append(y[i])

        x_tmp = np.array(x_tmp, float)
        y_tmp = np.array(y_tmp, int)

        x_tmp = np.reshape(x_tmp,(len(x_tmp),3,32,32))
        x_tmp = np.transpose(x_tmp,(0,3,1,2))
        x_tmp = np.transpose(x_tmp,(0,1,3,2))

        x_tmp = x_tmp*2./255. - 1.

        x_test = np.zeros((len(labels),int(size/len(labels)/10),32,32,3), dtype=float)

        pos = np.zeros(len(labels), dtype=int)

        for i in range(len(x_tmp)):
            x_test[y_tmp[i]][pos[y_tmp[i]]] = x_tmp[i]
            pos[y_tmp[i]] += 1

        self.data["test"] = x_test
        self.n_classes, self.n_examples, self.w, self.h, self.c = self.data["train"].shape
        self.n_classes, self.n_test_ex, self.w, self.h, self.c = self.data["test"].shape

    def get_batch(self,n,s="train"):
        """
        Create batch of n pairs
        """

        assert s in ["train","test"], "Choose either train or test"

        x = self.data[s]
        C1 = [random.choice(range(self.n_classes)) for i in range(n)]
        pairs = [np.zeros((n, self.h, self.w,self.c)) for i in range(2)]
        targets = np.zeros(n)
        targets[:n//2] = 1

        ##
        ## Pairs with the same class
        ##

        for i in range(n//2):
            R1 = random.randint(0,self.n_examples-1)
            R2 = random.randint(0,self.n_examples-1)


            # plt.imshow(x[C1[i]][R1])
            # plt.show()
            # plt.imshow(x[C1[i]][R2])
            # plt.show()

            pairs[0][i] = x[C1[i]][R1]
            pairs[1][i] = x[C1[i]][R2]

        ##
        ## Pairs with different classes
        ##

        for i in range(n//2,n):
            C2 = (random.randint(1,self.n_classes-1) + C1[i]) % self.n_classes
            R1 = random.randint(0,self.n_examples-1)
            R2 = random.randint(0,self.n_examples-1)

            pairs[0][i] = x[C1[i]][R1]
            pairs[1][i] = x[C2][R2]

        ##
        ## Shuffle pairs and targets
        ##

        perm = np.random.permutation(range(n))
        pairs[0] = pairs[0][perm]
        pairs[1] = pairs[1][perm]
        targets = targets[perm]

        return (pairs,targets)

    def get_validation_batch(self,n,s="test"):

        x = self.data[s]
        n_ex = x.shape[1]
        C1 = [random.choice(range(self.n_classes)) for i in range(n)]
        C2 = [random.choice(range(self.n_classes)) for i in range(n)]
        R1 = [random.choice(range(n_ex)) for i in range(n)]
        R2 = [random.choice(range(n_ex)) for i in range(n)]

        pairs = [np.zeros((n, self.h, self.w, self.c)) for i in range(2)]

        for i in range(n):
            pairs[0][i] = x[C1[i]][R1[i]]
            pairs[1][i] = x[C2[i]][R2[i]]

        target = np.equal(C1,C2)
        target = np.array(target, dtype=int)

        return (pairs, target)


    def PrintShape(self):
        print("Number of classes: %d" %self.n_classes)
        print("Number of training instances per class: %d" %self.n_examples)
        print("Number of test instances per class: %d" %self.n_test_ex)
        print("Image shape: %d,%d,%d" %(self.w, self.h, self.c))
