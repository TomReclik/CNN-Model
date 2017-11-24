import utils
import InceptionV3
import models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

#lang = {"Martensite":0, "Interface":1, "Boundary":2, "Evolved":3}
lang = {"Martensite":0, "Boundary":1, "Interface":2}

print "Loading Data"

loader = utils.SEM_loader(lang,[200,200],"/home/tom/Data/LabeledDamages/")

x_train, y_train, x_test, y_test = loader.getData(0.2)

print x_train.shape[0]+x_test.shape[0]

name = "InceptionV3"
weights_path = name + ".hdf5"
batch_size = 10

print "Initializing Network"

# network = models.EERACN(x_train, len(lang))

network = InceptionV3.InceptionV3(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=len(lang))

# network = models.EERACN(x_train, y_train, x_test, y_test, len(lang))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # keras.callbacks.TensorBoard(log_dir=('logs/EERACN_LBFGS_'+str(len(x_train))),
        #          histogram_freq=1,
        #          write_graph=False,
        #          write_images=False)
    ]

#network.fit(x_train, y_train, batch_size=batch_size, epochs=150, callbacks = callbacks, validation_split=0.2)

network.load_weights(weights_path)

score = network.evaluate(x_test, y_test, batch_size=batch_size)

print score
