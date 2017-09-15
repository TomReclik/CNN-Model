from __future__ import print_function, division
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

def conv_layer(x, filters, kernel_size, padding="valid", layer_name, seed=3, activation_function=tf.nn.LeakyReLU(0.18)):
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weights'):
            W = tf.get_variable('W', [input_dim, output_dim],
                                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        with tf.variable_scope('bias'):
            b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.1))
        with tf.variable_scope('output'):
            o = activation_function(tf.nn.xw_plus_b(x, W, b))
    return o

conv1 = tf.layers.conv2d(
    inputs=x_train,
    filters=192,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.LeakyReLU(0.18)
)

conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=160,
    kernel_size=[1,1],
    padding="valid",
    activation=tf.nn.LeakyReLU(0.18)
)

conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=96,
    kernel_size=[1,1],
    padding="valid",
    activation=tf.nn.LeakyReLU(0.18)
)

pool1 = tf.layers.nax_pooling2d(
    inputs=conv3,
    pool_size=[3,3],
    strides=2
)

dropout1 = tf.layers.dropout(
    inputs=pool1,
    rate=0.5
)

conv4 = tf.layers.conv2d(
    inputs=dropout1,
    filters=192,
    kernel_size=[5,5],
    padding="valid",
    activation=tf.nn.LeakyReLU(0.18)
)

conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=160,
    kernel_size=[1,1],
    padding="valid",
    activation=tf.nn.LeakyReLU(0.18)
)

conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=96,
    kernel_size=[1,1],
    padding="valid",
    activation=tf.nn.LeakyReLU(0.18)
)

pool1 = tf.layers.nax_pooling2d(
    inputs=conv3,
    pool_size=[3,3],
    strides=2
)

dropout1 = tf.layers.dropout(
    inputs=pool1,
    rate=0.5
)

model.add(Conv2D(192, (5,5), kernel_initializer='he_normal',
            bias_initializer='zeros'))
model.add(advanced_activations.LeakyReLU(alpha=0.18))
model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
            bias_initializer='zeros'))
model.add(advanced_activations.LeakyReLU(alpha=0.18))
model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
            bias_initializer='zeros'))
model.add(advanced_activations.LeakyReLU(alpha=0.18))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(192, (3,3), kernel_initializer='he_normal',
            bias_initializer='zeros'))
model.add(advanced_activations.LeakyReLU(alpha=0.18))
model.add(Conv2D(192, (1,1), kernel_initializer='he_normal',
            bias_initializer='zeros'))
model.add(advanced_activations.LeakyReLU(alpha=0.18))
model.add(Conv2D(10, (1,1), kernel_initializer='he_normal',
            bias_initializer='zeros'))
model.add(advanced_activations.LeakyReLU(alpha=0.18))

model.add(AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='same'))

model.add(Flatten())

model.add(Dense(NOL, activation='softmax'))

#
# Adding L-BFGS
#

loss = keras.losses.categorical_crossentropy()

tfoptimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            method='L-BFGS-B',
            options={'maxiter': iterations})

optimizer = keras.optimizers.TFOptimizer(tfoptimizer)

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, verbose=0),
    keras.callbacks.TensorBoard(log_dir=('logs/EERACN_LBFGS_'+str(len(x_train))),
             histogram_freq=1,
             write_graph=False,
             write_images=False)
]
model.fit(x_train, y_train, batch_size=20, epochs=150, callbacks = callbacks, validation_split=0.2)
score = model.evaluate(x_test, y_test, batch_size=20)

return score
