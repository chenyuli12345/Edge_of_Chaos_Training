import os #os模块提供了许多与操作系统交互的函数。例如可以使用os模块来读取环境变量、操作路径、创建或删除目录、获取当前工作目录等等。
import pickle #pickle模块提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。
from collections import OrderedDict #OrderedDict是一个字典子类，它记住了元素的添加顺序并提供了一些额外的方法

import numpy as np
from numpy import linalg as LA #numpy中的一个子模块，提供了线性代数的功能
from scipy.sparse.linalg import eigs #该子模块提供了一些稀疏线性代数功能，eigs函数则用于计算一个稀疏矩阵的一些特征值和特征向量
import tensorflow as tf
from tensorflow import keras


class LoggingReporter(keras.callbacks.Callback):
    """Save the activations to files at after some epochs.

    Args:
        args: configuration options dictionary
        x_test: test data
        y_test: test label
    """

    def __init__(self, args, x_train, y_train, x_test, y_test, *kargs, **kwargs): #这个类包含的第一个方法__init__，这是一个特殊的方法，也就是这个类的构造函数，用于初始化新创建的对象，接受了几个参数，其中最后两个参数是可变数量的参数，可以接收任意数量和类型的参数
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        #将传入的参数保存到类的实例变量中，以便在类的其他方法中使用
        self.args = args
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs=None):
        if not os.path.exists(self.args.save_weights_dir):
            os.makedirs(self.args.save_weights_dir)
        if not os.path.exists(self.args.save_losses_dir):
            os.makedirs(self.args.save_losses_dir)
        if not os.path.exists(self.args.save_scores_dir):
            os.makedirs(self.args.save_scores_dir)
        self.losses = {}
        self.losses['train'] = []
        self.losses['val'] = []
        self.W0_init = self.model.get_weights()[0]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.args.log_epochs:
            # Compute weights
            weights = save_weights(self.args, self.W0_init, self.model)
            # Save the weights
            fname = '{}/epoch{:05d}'.format(self.args.save_weights_dir, epoch)
            with open(fname, 'wb') as f:
                pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
        if epoch == 0:
            self.losses['train'].append(self.model.evaluate(self.x_train,
                                                            self.y_train,
                                                            verbose=0))
            self.losses['val'].append(self.model.evaluate(self.x_test,
                                                          self.y_test,
                                                          verbose=0))

    def on_epoch_end(self, epoch, logs=None):
        # print(logs.keys())
        self.losses['train'].append(
            [logs['loss'], logs['sparse_categorical_crossentropy'], logs['accuracy']])
        self.losses['val'].append(
            [logs['val_loss'], logs['val_sparse_categorical_crossentropy'], logs['val_accuracy']])
        # self.losses['train'].append(logs)
        # self.losses['val'].append(logs)

    def on_train_end(self, logs=None):
        # save training losses to file
        fname = '{}/losses'.format(self.args.save_losses_dir)
        with open(fname, 'wb') as f:
            pickle.dump(self.losses, f, pickle.HIGHEST_PROTOCOL)

        # save scores to file
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        fname = '{}/scores'.format(self.args.save_scores_dir)
        with open(fname, 'wb') as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)


def save_weights(args, W0_init, model):
    results = {}
    # W0 is 1st layer weights matrix
    W0 = model.get_weights()[0]
    W_diff = np.linalg.norm(W0-W0_init)
    # print('shape of difference is {}'.format((W0-W0_init).shape))
    results['W0s'] = W0
    results['W_diffs'] = W_diff

    return results
