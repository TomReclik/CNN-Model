import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics

def CCIFC(x_train, y_train, x_test, y_test):
    """
    Convolutional Network with:
        - 2 Convolutional layers
        - 1 Inception layer
        - 1 Fully connected output layer
    """

    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(16, (3, 3), activation='relu')(inputs)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(16, (4, 4), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.35)(inception)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs= Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=40, epochs=100)

    score = model.evaluate(x_test, y_test, batch_size=500)

    return score

def TIOSM(x_train, y_train, x_test, y_test):
    """
    Convolutional Network with:
        - 3 Inception layer
        - 1 Fully connected softmax output layer
    """

    inputs = Input(shape=(32, 32, 3))

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(10, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(64, (8, 8), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Flatten()(x)
    outputs= Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=40, epochs=20)

    score = model.evaluate(x_test, y_test, batch_size=40)

    return score

def EERACN(x_train, y_train, x_test, y_test):
    """
    Convolutional Network like in Empirical Evaluation of Rectified Activations in Convolution
Network
    """

    inputs = Input(shape=(32, 32, 3))

    tower_1 = Conv2D(100, (1, 1), padding='same', activation='relu')(inputs)
    tower_2 = Conv2D(100, (3, 3), padding='same', activation='relu')(inputs)
    tower_3 = Conv2D(100, (5, 5), padding='same', activation='relu')(inputs)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    tower_1 = Conv2D(100, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(100, (3, 3), padding='same', activation='relu')(x)
    tower_3 = Conv2D(100, (5, 5), padding='same', activation='relu')(x)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Dropout(0.5)(inception)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    tower_1 = Conv2D(100, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(100, (3, 3), padding='same', activation='relu')(x)
    tower_3 = Conv2D(100, (8, 8), padding='same', activation='relu')(x)

    inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Flatten()(x)
    outputs= Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=40, epochs=70)

    score = model.evaluate(x_test, y_test, batch_size=40)

    return score

def SCFN(x_train, y_train, x_test, y_test, channels, dropout, dense):
    """
    Convolutional Network with:
        - 6 Normal convolutional layers
        - 1 Fully connected output layer
    """

    model = Sequential()

    model.add(Conv2D(channels[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(channels[0], (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[0]))

    model.add(Conv2D(channels[1], (3, 3), activation='relu'))
    model.add(Conv2D(channels[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[1]))

    model.add(Conv2D(channels[2], (3, 3), activation='relu'))
    model.add(Conv2D(channels[2], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[2]))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(dropout[3]))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def FCFN(x_train, y_train, x_test, y_test):
    """
    Convolutional Network with:
        - 4 Normal convolutional layers
        - 1 Fully connected output layer
    """

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score

def TCFN(x_train, y_train, x_test, y_test, channels):
    """
    Convolutional Network with:
        - 4 Normal convolutional layers
        - 1 Fully connected output layer
    """

    model = Sequential()

    model.add(Conv2D(channels[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(channels[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=50, epochs=25)
    score = model.evaluate(x_test, y_test, batch_size=32)

    return score
