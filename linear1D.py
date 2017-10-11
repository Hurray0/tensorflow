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

def make_data(len=300):
    """伪造一维线性数据"""
    def get_shake():
        shake = k * (random() - 0.5) * (random() - 0.5) / 5
        return shake
    X = [random() for i in xrange(len)]
    k = random() * 10 - 5
    b = random() * 10 - 5
    Y = [k * x + b + get_shake() for x in X]
    return {'X': X, 'Y': Y, 'k': k, 'b': b}

def main(X, Y, show_plot=False):
    """一维线性预测
        Input: X, Y, show_plot(option)
        Output: [w,b] (Y = w * X + b)
    """
    W = tf.Variable(tf.random_uniform([1], -5, 5))
    b = tf.Variable(tf.zeros([1]))
    y = W * X + b

    loss = tf.reduce_mean(tf.square(y - Y))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for step in xrange(100):
        sess.run(train)
        print(step, sess.run(W), sess.run(b))
        print(step, sess.run(loss))
        #plt.plot(X, Y, 'ro')
        #plt.plot(X, sess.run(W) * X + sess.run(b), 'b')
        #plt.legend()
        #plt.show()

    if show_plot:
        plt.plot(X, Y, 'ro')
        plt.plot(X, sess.run(W) * X + sess.run(b), 'b')
        plt.legend()
        plt.show()

    return [sess.run(W)[0], sess.run(b)[0]]

if __name__ == '__main__':
    print '----------------------------------------'
    print '数据生成'
    data = make_data()
    print '数据生成成功'

    X = data['X']
    Y = data['Y']
    plt.plot(X, Y, 'o')
    result = main(X, Y, True)
    print 'origin: k=%s, b=%s' % (data['k'], data['b'])
    print 'predict: k=%s, b=%s' % (result[0], result[1])
    #plt.show()
