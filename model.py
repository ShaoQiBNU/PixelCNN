################################## load packages ###############################
import tensorflow as tf
from layer import *


################################## PixelCNN ###############################
class PixelCNN(object):
    def __init__(self, conf, x):

        ########## parameter ##########
        self.epochs = conf.epochs
        self.batch_size = conf.batch_size
        self.learning_rate = conf.learning_rate
        self.display_step = conf.display_step

        ########## data ##########
        self.q_levels = conf.q_levels
        self.classes = conf.classes
        self.height = conf.height
        self.width = conf.width
        self.channel = conf.channel

        ############# network ############
        self.gated_layers = conf.gated_layers
        self.gated_feature_maps = conf.gated_feature_maps
        self.output_conv_feature_maps = conf.output_conv_feature_maps


        ########## conv 7x7 mask A ##########
        net = conv(x, self.gated_feature_maps, [7, 7], 'A', scope="CONV_IN")


        ########## gated conv 3x3 mask B ##########
        for idx in range(self.gated_layers):
            scope = 'GATED_CONV%d' % idx
            net = gated_conv(net, [3, 3], scope=scope)


        ########## conv 1x1  ##########
        net = tf.nn.relu(conv(net, self.output_conv_feature_maps, [1, 1], "B", scope='CONV_OUT0'))

        ########## conv 1x1 shape [N,H,W,DC] ##########
        net = tf.nn.relu(conv(net, self.q_levels * self.channel, [1, 1], "B", scope='CONV_OUT1'))

        ########## reshape shape [N,H,W,DC] -> [N,H,W,D,C] ##########
        net = tf.reshape(net, [-1, self.height, self.width, self.q_levels, self.channel])

        ########## shape [N,H,W,D,C] -> [N,H,W,C,D] ##########
        net = tf.transpose(net, perm=[0, 1, 2, 4, 3])

        ########## [N,H,W,C,D] -> [NHWC,D] ##########
        net = tf.reshape(net, [-1, self.q_levels])

        self.pred = net