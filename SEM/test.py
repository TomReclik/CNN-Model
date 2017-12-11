import utils
import InceptionV3
import models
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

#lang = {"Martensite":0, "Interface":1, "Boundary":2, "Evolved":3}
# lang = {"Martensite":0, "Interface":1, "Evolved":2, "Evovled":2, "Notch":2}
# lang = {"Martensite":0, "Interface":1}
# lang = {"Notch":0,"Interface":1}
lang = {"Inclusion":0, "Notch":1, "Interface":1, "Martensite":1, "Evolved":1, "Boundary":1}
class_weight = {0:1.5,1:1}
dist = {"Inclusion":383, "Notch":100, "Interface":100, "Martensite":100, "Evolved":100, "Boundary":100}

compressedLang = []
for key,value in lang.iteritems():
    compressedLang.append(value)
compressedLang = set(compressedLang)
NOC = len(compressedLang)

name = "EERACN_IncVsAll_NoInc"
weights_path = name + ".hdf5"
batch_size = 5

print "Loading Data"

loader = utils.SEM_loader([250,250],"/home/tom/Data/LabeledDamages/")

x_train, y_train, x_test, y_test = loader.getData(lang,0.2,dist=dist)

print "Initializing Network"

network = models.EERACN(x_train, NOC)

# network = InceptionV3.InceptionV3(include_top=True,
#                 weights=None,
#                 input_tensor=None,
#                 input_shape=None,
#                 pooling=None,
#                 classes=NOC)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, verbose=0),
        ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # keras.callbacks.TensorBoard(log_dir=('logs/EERACN_LBFGS_'+str(len(x_train))),
        #          histogram_freq=1,
        #          write_graph=False,
        #          write_images=False)
    ]

#network.fit(x_train, y_train, batch_size=batch_size, epochs=150,
#            callbacks = callbacks, validation_split=0.2, class_weight=class_weight)

network.load_weights(weights_path)

score = network.evaluate(x_test, y_test, batch_size=batch_size)

print "Accuracy on the entire test batch: ", score[1]

##
## Test accuracy on the subclasses
##

inv_lang = {v: k for k,v in lang.iteritems()}

x = [[] for i in range(NOC)]
y = [[] for i in range(NOC)]

for i in range(y_test.shape[0]):
    for j in range(NOC):
        if y_test[i][j] == 1:
            x[j].append(x_test[i])
            y[j].append(j)

for i in range(NOC):
    x_tmp = np.asarray(x[i], float)
    y_tmp = np.asarray(y[i], int)
    y_tmp = keras.utils.to_categorical(y_tmp, NOC)
    score = network.evaluate(x_tmp, y_tmp, batch_size=batch_size, verbose=0)
    print "Accuracy on just the ", inv_lang[i], "test data", score[1]
    print x_tmp.shape

#
# M = []
# y_M = []
#
# ID = []
# y_ID = []
#
# # N = []
# # y_N = []
#
# # B = []
# # y_B = []
# #
#
# for i in range(y_test.shape[0]):
#     if y_test[i][0] == 1:
#         M.append(x_test[i])
#         y_M.append(0)
#     elif y_test[i][1] == 1:
#         ID.append(x_test[i])
#         y_ID.append(1)
#     # elif y_test[i][2] == 1:
#     #     N.append(x_test[i])
#     #     y_N.append(2)
#     # elif y_test[i][2] == 1:
#     #     B.append(x_test[i])
#     #     y_B.append(2)
#
#     # else:
#     #     I.append(x_test[i])
#     #     y_I.append(2)
#
# M = np.asarray(M, float)
# y_M = np.asarray(y_M, int)
# y_M = keras.utils.to_categorical(y_M, NOC)
# score_M = network.evaluate(M, y_M, batch_size=batch_size, verbose=0)
# print "Accuracy on just the Interface test data: ", score_M[1]
#
# ID = np.asarray(ID, float)
# y_ID = np.asarray(y_ID, int)
# y_ID = keras.utils.to_categorical(y_ID, NOC)
# score_ID = network.evaluate(ID, y_ID, batch_size=batch_size, verbose=0)
# print "Accuracy on just the Notch test data: ", score_ID[1]
#
# # B = np.asarray(B, float)
# # y_B = np.asarray(y_B, int)
# # y_B = keras.utils.to_categorical(y_B, NOC)
# # score_B = network.evaluate(B, y_B, batch_size=batch_size, verbose=0)
# # print "Accuracy on just the Boundary test data: ", score_B[1]
# #
# # N = np.asarray(N, float)
# # y_N = np.asarray(y_N, int)
# # y_N = keras.utils.to_categorical(y_N, NOC)
# # score_N = network.evaluate(N, y_N, batch_size=batch_size, verbose=0)
# # print "Accuracy on just the Notch test data: ", score_N[1]
#
# # I = np.asarray(I, float)
# # y_I = np.asarray(y_I, int)
# # y_I = keras.utils.to_categorical(y_I, NOC)
# # score_I = network.evaluate(I, y_I, batch_size=batch_size)
# # print "Accuracy on just the Inclusion test data: ", score_I[1]
