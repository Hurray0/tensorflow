#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author: Hurray(hurray0@icloud.com)
# Date: 2017.10.10 14:27:11

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # 可视化模块

from random import random

def make_data(dimension=5, len=300):
    """伪造多维线性数据"""
    def get_shake():
        shake = (random() - 0.5) * (random() - 0.5)
        return shake
    X = [[random() for i in xrange(len)] for d in xrange(dimension)]
    K = [random() for d in xrange(dimension)]
    b = random()
    shakes = [get_shake() for i in xrange(len)]

    Y2 = np.dot(K, X) + b
    Y = np.dot(K, X) + b + shakes
    return {'X': X, 'Y': Y, 'Y2': Y2, 'K': K, 'b': b}

def main(X, Y):
    """多维线性预测
        Input: X, Y, show_plot(option)
        Output: [w,b] (Y = w * X + b)
    """
    W = tf.Variable(tf.random_uniform([1, len(X)], 0.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(W, X) + b

    loss = tf.reduce_mean(tf.square(y - Y))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for step in xrange(1000):
        sess.run(train)
        if step % 100 == 0:
            print(step, sess.run(W), sess.run(b))
            print(step, sess.run(loss))

    return [sess.run(W), sess.run(b)[0]]

if __name__ == '__main__':
    print '----------------------------------------'
    print '数据生成'
    data = make_data(dimension = 5)
    print '数据生成成功'

    X = data['X']
    Y = data['Y']
    #plt.plot(X, Y, 'o')
    #plt.show()

    #print data
    result = main(X, Y)
    print 'origin: k=%s, b=%s' % (data['K'], data['b'])
    print 'predict: k=%s, b=%s' % (result[0][0], result[1])
