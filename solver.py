################################## load packages ###############################
import tensorflow as tf
import numpy as np
from model import PixelCNN
import matplotlib.pyplot as plt


################################## slover ###############################
class Solver(object):
    def __init__(self, conf, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.conf = conf

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

        ########## sample ##########
        self.show_figure = conf.show_figure
        self.figure = conf.figure

        ########## placeholder ##########
        self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.y = tf.placeholder(tf.int64, [None, self.height, self.width, self.channel])
        self.y = tf.reshape(self.y, [-1])

        #### model pred 影像判断结果 ####
        self.pred = PixelCNN(self.conf, self.x).pred

        #### loss 损失计算 ####
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        #### optimization 优化 ####
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        #### accuracy 准确率 ####
        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.pred), 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    #################### train ##################
    def train(self):
        ########## initialize variables ##########
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            step = 1

            #### epoch 世代循环 ####
            for epoch in range(self.epochs + 1):

                #### iteration ####
                for _ in range(len(self.X_train) // self.batch_size):

                    step += 1

                    ##### get x,y #####
                    batch_x, batch_y = self.random_batch(self.X_train, self.batch_size, self.q_levels)

                    ##### optimizer ####
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                    ##### show loss and acc #####
                    if step % self.display_step == 0:
                        loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y})
                        print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))

            print("Optimizer Finished!")


    #################### test ##################
    def test(self):
        ########## initialize variables ##########
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            #### iteration ####
            for _ in range(len(self.X_test) // self.batch_size):
                ##### get x,y #####
                batch_x, batch_y = self.random_batch(self.X_test, self.batch_size, self.q_levels)

                ##### show loss and acc #####
                loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y})
                print(", Minibatch Loss=" + "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            print("Test Finished!")


    ################## 生成sample ##################
    def generate_sample(self):
        for _ in range(self.show_figure):

            ##### get x,y #####
            samples, _ = self.random_batch(self.X_test, self.figure, self.q_levels)

            ##### occlude start row #####
            occlude_start_row = 18

            ##### occlude start row 之下的设置为0 #####
            samples[:, occlude_start_row:, :, :] = 0.

            ##### 逐像元预测 #####
            for i in range(occlude_start_row, self.height):
                for j in range(self.width):
                    for k in range(self.channel):
                        next_sample = self.predict(samples) / (self.q_levels - 1.)
                        samples[:, i, j, k] = next_sample[:, i, j, k]

            ##### 效果展示 #####
            plt.figure()
            for i in range(self.figure):
                plt.subplot(self.figure // 4, 4, i)
                plt.imshow(samples[i, :, :, :])
            plt.show()


    ################## 预测结果 ##################
    def predict(self, samples):
        '''
        :param samples: [N, H, W, C]
        :return: [N, H, W, C]
        '''

        ########## initialize variables ##########
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            ########## shape [N,H,W,C,D], values [probability distribution] ##########
            pixel_value_probabilities = sess.run(self.pred, feed_dict={self.x: samples})

            ########## shape [N,H,W,C], values [index of most likely pixel value] ##########
            pixel_value_indices = np.argmax(pixel_value_probabilities, 4)

        return pixel_value_indices


    ################### random generate data ###################
    def random_batch(self, images, batch_size, levels):
        '''
        :param images: 输入影像集
        :return: batch data
                 label: 影像集的label
                 输出size [N,H,W,C]
        '''

        num_images = len(images)

        ######## 随机设定待选图片的id ########
        idx = np.random.choice(num_images, size=batch_size, replace=False)

        ######## 筛选data ########
        x_batch = images[idx, :, :]

        ######## label ########
        y_batch = np.reshape(np.clip(((x_batch * levels).astype('int64')), 0, levels - 1), [-1,]) # [N,H,W,C] -> [NHWC]

        return x_batch, y_batch