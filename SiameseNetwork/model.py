def EERACN(x_train, y_train, x_test, y_test, NOL):
    """
    Network proposed in the paper by Bing Xu et. al.
    https://arxiv.org/pdf/1505.00853.pdf
    """

    model = Sequential()

    model.add(Conv2D(192, (5,5), input_shape=(32,32,3), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(160, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(96, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    #model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (5,5), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3,3), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())
    model.add(Conv2D(10, (1,1), kernel_initializer='he_normal',
                bias_initializer='zeros'))

    model.add(BatchNormalization(axis=3))

    # model.add(advanced_activations.LeakyReLU(alpha=0.31))
    model.add(advanced_activations.ELU())

    model.add(AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='same'))

    model.add(Flatten())

    model.add(Dense(NOL, activation='softmax'))
