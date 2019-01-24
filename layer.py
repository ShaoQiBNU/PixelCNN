################################## load packages ###############################
import tensorflow as tf
import numpy as np


################################## mask ###############################
def create_mask(weights_shape, mask_type):
    # mask type
    #         -------------------------------------
    #        |  1       1       1       1       1 |
    #        |  1       1       1       1       1 |
    #        |  1       1    1 if B     0       0 |   H // 2
    #        |  0       0       0       0       0 |   H // 2 + 1
    #        |  0       0       0       0       0 |
    #         -------------------------------------
    #  index    0       1     W//2    W//2+1


    ########## weights shape ##########
    kernel_h, kernel_w, num_inputs, features = weights_shape

    ########## kernel的中心点 ##########
    center_h, center_w = kernel_h // 2, kernel_w // 2

    ########## mask ##########
    mask = np.ones((kernel_h, kernel_w, num_inputs, features), dtype=np.float32)

    ########## 垂直mask ##########
    if mask_type == 'V':
        mask[center_h:, :, :, :] = 0.

    ########## 水平mask ##########
    else:

        ########## mask B ##########
        mask[center_h, center_w + 1:, :, :] = 0.
        mask[center_h + 1:, :, :, :] = 0.

        ########## mask A ##########
        if mask_type == 'A':
            mask[center_h, center_w:, :, :] = 0.

    return mask


################################## weights and biases ###############################
def get_weights(shape, name, scope):
    with tf.variable_scope(scope):
        weights_initializer = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name, shape, tf.float32, weights_initializer)
    return W

def get_biases(shape, name, scope):
    with tf.variable_scope(scope):
        b = tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))
    return b


################################## convlution ###############################
def conv2d(x, W, b, strides=1, padding='SAME'):
	x=tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
	x=tf.nn.bias_add(x, b)
	return x


def conv(inputs, features, kernel, mask_type, scope, strides=1, padding='SAME'):

    ########## inputs、kernel、weights and biases shape ##########
    num_inputs = inputs.get_shape().as_list()[-1]

    kernel_h, kernel_w = kernel
    weights_shape = [kernel_h, kernel_w, num_inputs, features]
    W = get_weights(weights_shape, 'weight', scope)

    biases_shape = [features, ]
    b = get_biases(biases_shape, 'bias', scope)

    ########## mask ##########
    mask = create_mask(weights_shape, mask_type)
    W *= mask

    ########## convlution ##########
    net = conv2d(inputs, W, b, strides=strides, padding=padding)

    return net


################################## gated convlution ###############################
def _gated_activation_unit(inputs, kernel, mask_type, scope):
    '''
    :return: [N,H,W,C[,D]]
    '''

    ########## 获取feature maps ##########
    p2 = inputs.get_shape().as_list()[-1]

    ########## blue diamond ##########
    ########## 2p in channels, 2p out channels, mask, same padding, stride 1 ##########
    bd_out = conv(inputs, p2, kernel, mask_type, scope)

    ########## split 2p out channels into p going to tanh and p going to sigmoid ##########
    bd_out_0, bd_out_1 = tf.split(bd_out, 2, 3)
    tanh_out = tf.tanh(bd_out_0)
    sigmoid_out = tf.sigmoid(bd_out_1)

    return tanh_out * sigmoid_out

def gated_conv(inputs, kernel, scope):

    with tf.variable_scope(scope):
        ########## 输入inputs拆成两部分 ##########
        horiz_inputs, vert_inputs = tf.split(inputs, 2, 3)

        ########## 获取feature maps ##########
        p = horiz_inputs.get_shape().as_list()[-1]
        p2 = 2 * p

        ########## vertical n x n conv ##########
        ########## p in channels, 2p out channels, vertical mask, same padding, stride 1 ##########
        vert_nxn = conv(vert_inputs, p2, kernel, 'V', scope="vertical_nxn")

        ########## vertical blue diamond ##########
        ########## 2p in channels, p out channels, vertical mask ##########
        vert_gated_out = _gated_activation_unit(vert_nxn, kernel, 'V', scope="vertical_gated_activation_unit")

        ########## vertical 1 x 1 conv ##########
        ########## 2p in channels, 2p out channels, no mask?, same padding, stride 1 ##########
        vert_1x1 = conv(vert_nxn, p2, [1, 1], 'V', scope="vertical_1x1")

        ########## horizontal 1 x n conv ##########
        ########## p in channels, 2p out channels, horizontal mask B, same padding, stride 1 ##########
        horiz_1xn = conv(horiz_inputs, p2, kernel, 'B', scope="horizontal_1xn")
        horiz_gated_in = vert_1x1 + horiz_1xn

        ########## horizontal blue diamond ##########
        ########## 2p in channels, p out channels, horizontal mask B ##########
        horiz_gated_out = _gated_activation_unit(horiz_gated_in, kernel, 'B', scope="horizontal_gated_activation_unit")

        ########## horizontal 1 x 1 conv ##########
        ########## p in channels, p out channels, mask B, same padding, stride 1 ##########
        horiz_1x1 = conv(horiz_gated_out, p, kernel, 'B', scope="horizontal_1x1")

        ########## add ##########
        horiz_outputs = horiz_1x1 + horiz_inputs

    return tf.concat([horiz_outputs, vert_gated_out], 3)