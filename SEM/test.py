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
lang = {"Evolved":0, "Evovled":0, "Notch":0, "Martensite":1, "Boundary":1, "Interface":1}


compressedLang = []
for key,value in lang.iteritems():
    compressedLang.append(value)
compressedLang = set(compressedLang)
NOC = len(compressedLang)

name = "EERACN_EvsAll_NoInc"
weights_path = name + ".hdf5"
batch_size = 10

print "Loading Data"

loader = utils.SEM_loader(lang,[200,200],"/home/tom/Data/LabeledDamages/")

x_train, y_train, x_test, y_test = loader.getData(0.2)

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

network.fit(x_train, y_train, batch_size=batch_size, epochs=150,
            callbacks = callbacks, validation_split=0.2, class_weight={0:1, 1:2})

network.load_weights(weights_path)

score = network.evaluate(x_test, y_test, batch_size=batch_size)

print "Accuracy on the entire test batch: ", score[1]

# pred = network.predict(x_test)

# TN = utils.TrueNegatives(pred,y_test,0.7)

# print TN, "of ", y_test.shape[0]



##
## Test accuracy on the subclasses
##

M = []
y_M = []

ID = []
y_ID = []

# I = []
# y_I= []

for i in range(y_test.shape[0]):
    if y_test[i][0] == 1:
        M.append(x_test[i])
        y_M.append(0)
    elif y_test[i][1] == 1:
        ID.append(x_test[i])
        y_ID.append(1)
    # else:
    #     I.append(x_test[i])
    #     y_I.append(2)

M = np.asarray(M, float)
y_M = np.asarray(y_M, int)
y_M = keras.utils.to_categorical(y_M, NOC)
score_M = network.evaluate(M, y_M, batch_size=batch_size)
print "Accuracy on just the evolved test data: ", score_M[1]

ID = np.asarray(ID, float)
y_ID = np.asarray(y_ID, int)
y_ID = keras.utils.to_categorical(y_ID, NOC)
score_ID = network.evaluate(ID, y_ID, batch_size=batch_size)
print "Accuracy on just the not evolved test data: ", score_ID[1]


# I = np.asarray(I, float)
# y_I = np.asarray(y_I, int)
# y_I = keras.utils.to_categorical(y_I, NOC)
# score_I = network.evaluate(I, y_I, batch_size=batch_size)
# print "Accuracy on just the Inclusion test data: ", score_I[1]
