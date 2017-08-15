import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.callbacks import EarlyStopping

def MaxPoolIncreasingChannel(x_train, y_train, x_test, y_test, NOL, k):
    """
    This function will train a network oriented around:
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    It uses an increasing number of channels in order to compensate for the
    decreasing size of the input from layer to layer
    """

    model = Sequential()

    #
    # 1x320
    #

    model.add(Conv2D(k, (2,2), input_shape=(32,32,3)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(k, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 2x320
    #

    model.add(Conv2D(2*k, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(Conv2D(2*k, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 3x320
    #

    model.add(Conv2D(3*k, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(Conv2D(3*k, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 4x320
    #

    model.add(Conv2D(4*k, (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Conv2D(4*k, (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=2),
        keras.callbacks.TensorBoard(log_dir='logs',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False)
    ]

    model.fit(x_train, y_train, batch_size=50, epochs=150, callbacks=callbacks, verbose=1, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def MaxPoolConstantChannel(x_train, y_train, x_test, y_test, NOL, k):
    """
    This function will train a network oriented around:
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    It uses the same number of channels in each layer while the picture is
    downsampled until it reaches the final layer
    """

    model = Sequential()

    #
    # 1x320
    #

    model.add(Conv2D(int(2.5*k), (2,2), input_shape=(32,32,3)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 2x320
    #

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 3x320
    #

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #
    # 6x320
    #

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Conv2D(int(2.5*k), (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=2),
        keras.callbacks.TensorBoard(log_dir='logs/MPCC',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False)
    ]

    model.fit(x_train, y_train, batch_size=50, epochs=150, callbacks=callbacks, verbose=1, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def NoMaxPoolConstantChannel(x_train, y_train, x_test, y_test, NOL, k):
    """
    This function will train a network oriented around:
    http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
    It uses the same number of channels in each layer with no downsampling
    """

    model = Sequential()

    #
    # 1x320
    #

    model.add(Conv2D(int(2.5*k), (2,2), input_shape=(32,32,3)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    #
    # 2x320
    #

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.1))

    #
    # 3x320
    #

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.25))

    #
    # 6x320
    #

    model.add(Conv2D(int(2.5*k), (2,2)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Conv2D(int(2.5*k), (1,1)))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, verbose=2),
        keras.callbacks.TensorBoard(log_dir='logs/NMPCC',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False)
    ]

    model.fit(x_train, y_train, batch_size=50, epochs=150, callbacks=callbacks, verbose=1, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score
