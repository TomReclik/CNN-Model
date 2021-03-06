import utils
import InceptionV3
import models
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import confusion_matrix

##
## Set seed
##

np.random.seed(42)

#lang = {"Martensite":0, "Interface":1, "Boundary":2, "Evolved":3}
# lang = {"Martensite":0, "Interface":1, "Evolved":2, "Evovled":2, "Notch":2}
# lang = {"Martensite":0, "Interface":1}
# lang = {"Notch":0,"Interface":1}
lang = { "Martensite":0, "Notch":1, "Interface":2, "Boundary":3}
class_weight = {0:1,1:1,2:1,3:1}
dist = { "Martensite":400, "Notch":400, "Interface":400, "Boundary":115}

compressedLang = []
for key,value in lang.iteritems():
    compressedLang.append(value)
compressedLang = set(compressedLang)
NOC = len(compressedLang)

name = "EERACN_SecondStage"
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

network.fit(x_train, y_train, batch_size=batch_size, epochs=150,
           callbacks = callbacks, validation_split=0.2, class_weight=class_weight)

network.load_weights(weights_path)

score = network.evaluate(x_test, y_test, batch_size=batch_size)

print "Accuracy on the entire test batch: ", score[1]

##
## Test accuracy on the subclassesskmetrics
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

y_pred = network.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cmatrix = confusion_matrix(y_true, y_pred)

print cmatrix
