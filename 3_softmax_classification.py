#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author: Hurray(hurray0@icloud.com)
# Date: 2017.10.11 16:26:26

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

def read_data_mnist():
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data",
            one_hot=True)
    return {'X': mnist.train.images,
            'Y': mnist.train.labels,
            'X2': mnist.test.images,
            'Y2': mnist.test.labels}

def get_data():
    return read_data_mnist()

def softmax(X, Y, X2, Y2):
    """softmax分类器"""
    dimension = len(X[0])
    class_num = len(Y[0])
    # Create the model
    x = tf.placeholder(tf.float32, [None, dimension])
    W = tf.Variable(tf.zeros([dimension, class_num]))
    b = tf.Variable(tf.zeros([class_num]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, class_num])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print('准备工作完成，开始训练')
    # Train
    for _ in range(300):
        print _
        sess.run(train_step, feed_dict={x: X, y_: Y})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy: %s" % sess.run(accuracy, feed_dict={x: X2,
                                        y_: Y2}))

if __name__ == '__main__':
    data = get_data()
    softmax(data['X'], data['Y'], data['X2'], data['Y2'])
