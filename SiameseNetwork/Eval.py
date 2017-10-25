import utils
import model
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(42)

labels = [0,1,2,3,4]
TRAININGSIZE = 1000
EPOCHS = 100
BATCHSIZE = 32
EVAL = 100

# (x_train,x_test) = utils.loadCIFAR10(TRAININGSIZE,labels,val_split=0.)
#
# x = [np.zeros((NOE,32,32,3)) for i in range(2)]
# y = np.zeros(NOE,dtype=int)
#
# for i in range(NOE):
#     C1 = np.random.randint(0, len(labels))
#     C2 = np.random.randint(0, len(labels))
#     R1 = np.random.randint(0,TRAININGSIZE/len(labels))
#     R2 = np.random.randint(0,TRAININGSIZE/len(labels))
#
#     x[0][i] = x_train[C1][R1]
#     x[1][i] = x_train[C2][R2]
#     y[i] = int(C1==C2)
#
# siamese = model.siamese_EERACN()
#
# siamese.fit(x, y, batch_size=32, epochs=150, validation_split=0.2)

loader = utils.Siamese_Loader(TRAININGSIZE,labels,os.getcwd() + "/../CIFAR-10")

siamese = model.siamese_EERACN()

(pairs, targets) = loader.get_validation_batch(10)
#
# for i in range(EPOCHS):
#     (pairs,targets) = loader.get_batch(BATCHSIZE)
#     siamese.train_on_batch(pairs, targets)
#
#
