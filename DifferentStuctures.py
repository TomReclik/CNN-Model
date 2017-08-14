import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.callbacks import EarlyStopping

def MaxPoolIncreasingChannel(x_train, y_train, x_test, y_test, NOL):
    """
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    without sparsity
    k = 320
    """

    model = Sequential()

    #
    # 1x320
    #

    model.add(Conv2D(320, (2,2), input_shape=(32,32,3)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(320, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 2x320
    #

    model.add(Conv2D(640, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(Conv2D(640, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 3x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 6x320
    #

    model.add(Conv2D(1280, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Conv2D(1280, (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=4, verbose=0),
        keras.callbacks.TensorBoard(log_dir='MPIC.log',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False)

    ]

    model.fit(x_train, y_train, batch_size=50, epochs=30, callbacks=callbacks)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def MaxPoolConstantChannel(x_train, y_train, x_test, y_test, NOL):
    """
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    without sparsity
    k = 320
    """

    model = Sequential()

    #
    # 1x320
    #

    model.add(Conv2D(960, (2,2), input_shape=(32,32,3)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 2x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 3x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 6x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Conv2D(960, (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=4, verbose=0),
        keras.callbacks.TensorBoard(log_dir='MPCC.log',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False)

    ]

    model.fit(x_train, y_train, batch_size=50, epochs=30,, callbacks=callbacks)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def NoMaxPoolConstantChannel(x_train, y_train, x_test, y_test, NOL):
    """
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    without sparsity
    k = 320
    """

    model = Sequential()

    #
    # 1x320
    #

    model.add(Conv2D(960, (2,2), input_shape=(32,32,3)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    #
    # 2x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    #
    # 3x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    #
    # 6x320
    #

    model.add(Conv2D(960, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Conv2D(960, (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=4, verbose=0),
        keras.callbacks.TensorBoard(log_dir='NMPCC.log',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False)

    ]

    model.fit(x_train, y_train, batch_size=50, epochs=30, callbacks=callbacks)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score
