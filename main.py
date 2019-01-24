################################## load packages ###############################
import tensorflow as tf
import numpy as np
import argparse
import cifar10
from solver import Solver


###################### load data #########################
def load_data():

    ################ download dataset ####################
    cifar10.maybe_download_and_extract()

    ################ load train and test data ####################
    images_train, _, _ = cifar10.load_training_data()
    images_test, _, _ = cifar10.load_test_data()

    return images_train, images_test


################################## main ###############################
if __name__ == "__main__":

    ###################### load train and test data #########################
    images_train, images_test = load_data()

    ###################### argument ####################
    parser = argparse.ArgumentParser()

    ############# parameter ##############
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--display_step', type=int, default=20)

    ############# data #############
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--channel', type=int, default=3)

    ############# network ############
    parser.add_argument('--gated_layers', type=int, default=15)
    parser.add_argument('--gated_feature_maps', type=int, default=256)
    parser.add_argument('--output_conv_feature_maps', type=int, default=1024)
    parser.add_argument('--q_levels', type=int, default=256)

    ############# sample ############
    parser.add_argument('--show_figure', type=int, default=2)
    parser.add_argument('--figure', type=int, default=16)

    ############# conf #############
    conf = parser.parse_args()


    ###################### solver ####################
    solver = Solver(conf, images_train, images_test)

    ############# train #############
    solver.train()

    ############# test #############
    solver.test()

    ############# sample #############
    solver.generate_sample()