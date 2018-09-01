import tensorflow as tf

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                         num_filters]

    #initialize weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03)
                          ,name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation

    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1,2,2,1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer



