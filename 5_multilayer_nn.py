#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author: Hurray(hurray0@icloud.com)
# Date: 2017.10.13 09:13:51

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import numpy as np
import tensorflow as tf
from random import random

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

def _random_batch(X, Y, num=50):
    size = len(X)
    if not num < size:
        return X, Y

    n = []
    for i in xrange(num):
        n += [int(random() * size)]

    return [X[i] for i in n], [Y[i] for i in n]

def simple_nn(X, Y, X2, Y2):
    """简单神经网络"""
    dimension = len(X[0])
    class_num = len(Y[0])

    x = tf.placeholder(tf.float32, [None, dimension])
    y_ = tf.placeholder(tf.float32, [None, class_num])

    with tf.name_scope('layer1'):
        W = tf.Variable(tf.random_normal([dimension, 32]))
        b = tf.Variable(tf.random_normal([32]))
        h = tf.nn.tanh(tf.matmul(x, W) + b)

    with tf.name_scope('layer2'):
        W2 = tf.Variable(tf.random_normal([32, 32]))
        b2 = tf.Variable(tf.random_normal([32]))
        h2 = tf.nn.tanh(tf.matmul(h, W2) + b2)

    with tf.name_scope('layer3'):
        W3 = tf.Variable(tf.random_normal([32, class_num]))
        b3 = tf.Variable(tf.random_normal([class_num]))
        y = tf.matmul(h2, W3) + b3

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(2).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print('准备工作完成，开始训练')
    # Train
    for _ in range(30000):
        _x, _y = _random_batch(X, Y, 200)
        sess.run(train_step, feed_dict={x: _x, y_: _y})
        if _ % 1000 == 0:
            print "%s loss: %s" % (_,
                    sess.run(cross_entropy, feed_dict={x: X, y_: Y}))
            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("accuracy: %s" % sess.run(accuracy, feed_dict={x: X2,
                                                y_: Y2}))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy: %s" % sess.run(accuracy, feed_dict={x: X2,
                                        y_: Y2}))

if __name__ == '__main__':
    data = get_data()
    simple_nn(data['X'], data['Y'], data['X2'], data['Y2'])
