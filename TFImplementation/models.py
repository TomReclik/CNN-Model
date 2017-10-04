import tensorflow as tf

def EERACN(x, NOL, dropout, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('EERACN'):
        # Convolution Layer with 192 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 192, 5,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        conv1 = tf.layers.conv2d(conv1, 160, 1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        conv1 = tf.layers.conv2d(conv1, 96, 1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        pool1 = tf.layers.max_pooling2d(conv1, 3, 2)

        conv1_drop = tf.nn.dropout(pool1, keep_dropout)

        # Convolution Layer with 192 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1_drop, 192, 5,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        conv2 = tf.layers.conv2d(conv2, 192, 1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        conv2 = tf.layers.conv2d(conv2, 192, 1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        pool2 = tf.layers.max_pooling2d(conv2, 3, 2)

        conv2_drop = tf.nn.dropout(pool2, keep_dropout)

        # Convolution Layer with 192 filters and a kernel size of 5
        conv3 = tf.layers.conv2d(conv2_drop, 192, 3,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        conv3 = tf.layers.conv2d(conv3, 192, 1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        conv3 = tf.layers.conv2d(conv3, 10, 1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer='zeros',activation=tf.nn.LeakyReLU(0.18))
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        pool3 = tf.layers.average_pooling2d(conv3, 8, 1, padding='same')

        flat = tf.contrib.layers.flatten(pool3)

        yhat = tf.layers.dense(flat,NOL,activation='softmax')

    return yhat
