import argparse #argparse是用来处理命令行参数的工具，提供了一种方便的方式来定义用户应该如何指定/处理这些参数
import time

import tensorflow as tf
from tensorflow import keras #keras是tf上的一个高级接口，用于构建和训练深度学习模型
from tensorflow.keras.losses import SparseCategoricalCrossentropy #这是一个损失函数，常用于多分类问题，是交叉熵损失函数的一种（适用于真实标签是整数的情况，函数会将真实标签视为类别索引，并使用这个索引从预测的概率分布中选择一个概率）
#例如，如果你有一个分类问题，类别有 3 类，那么你的真实标签可能是 0、1 或 2。你的模型会预测出一个概率分布，如 [0.1, 0.2, 0.7]，这表示模型认为样本属于第 0 类、第 1 类和第 2 类的概率分别是 0.1、0.2 和 0.7。
import numpy as np

#下面的包均为此目录下的程序
import utils
import loggingreporter
import plot_figure2
import plot_figure4_5
import plot_figure6

# Training settings
parser = argparse.ArgumentParser(
    description='Asymptotic stability study of SGD')
parser.add_argument('--dataset', type=str, default='mnist',
                    help="datset {'mnist', 'kmnist', 'emnist/mnist'}. default: 'mnist'")
parser.add_argument('--activation-func', type=str, default='relu',
                    help='activation function for hidden layers')
parser.add_argument('--epochs', default=4, type=int, metavar='N',
                    help='number of total epochs to run, should > 3')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimizer used for training')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD')
parser.add_argument('--beta-1', default=0.9, type=float, metavar='M',
                    help='beta_1 in Adam')
parser.add_argument('--beta-2', default=0.999, type=float, metavar='M',
                    help='beta_2 in Adam')
parser.add_argument('--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--num-repeats', type=int, default=10,
                    help='number of simulation repeats')


def main():
    start_time = time.time()
    args = parser.parse_args()
    args.log_epochs = np.arange(args.epochs)
    args.arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
        args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(args.weight_decay*1000000))
    args.save_weights_dir = 'rawdata/weights/{}'.format(args.arguments)
    args.save_losses_dir = 'rawdata/losses/{}'.format(args.arguments)
    args.save_scores_dir = 'rawdata/scores/{}'.format(args.arguments)

    # (x_train, y_train), (x_test, y_test) = utils.load_qmnist_data()
    (x_train, y_train), (x_test, y_test) = utils.load_data(args.dataset)
    args.input_shape = x_train.shape[1]

    for num_repeat in range(args.num_repeats):
        # break
        print('num_repeat={}'.format(num_repeat))
        args.save_weights_dir = 'rawdata/weights/{}/{}'.format(
            args.arguments, num_repeat)
        args.save_losses_dir = 'rawdata/losses/{}/{}'.format(
            args.arguments, num_repeat)
        args.save_scores_dir = 'rawdata/scores/{}/{}'.format(
            args.arguments, num_repeat)

        if args.activation_func == 'relu':
            activation_func = tf.nn.relu
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(args.input_shape, activation=activation_func, name='layer_1',
                                      use_bias=False, input_shape=(args.input_shape,),
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay),),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay))
            ])
        elif args.activation_func == 'tanh':
            activation_func = tf.nn.tanh
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(args.input_shape, activation=activation_func, name='layer_1',
                                      kernel_initializer=tf.keras.initializers.RandomNormal(
                                          mean=(0)/(args.input_shape), stddev=1/(2*np.sqrt(args.input_shape))),
                                      use_bias=False, input_shape=(args.input_shape,),
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay),),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      # kernel_initializer=tf.keras.initializers.RandomNormal(
                                      #     mean=(0)/(args.input_shape), stddev=1/(2*np.sqrt(args.input_shape))),
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay))
            ])

        if args.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=args.lr, momentum=args.momentum, nesterov=False, name='SGD')
        elif args.optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=1e-07,
                amsgrad=False, name='Adam')
        metric_loss = SparseCategoricalCrossentropy(from_logits=False,
                                                    name='sparse_categorical_crossentropy')
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=[metric_loss, 'accuracy'])
        reporter = loggingreporter.LoggingReporter(
            args, x_train, y_train, x_test, y_test)
        # model.fit(x_train, y_train, epochs=args.epochs,
        #           verbose=0, callbacks=[reporter, ], validation_split=0.2)
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                  verbose=0, callbacks=[reporter, ], validation_data=(x_test, y_test))
        tf.keras.backend.clear_session()

    # Plot Fig.2 and Fig.3(a) in the paper.
    # For Fig.3(b)-(d), you need to record the slope at Fig.3(a) for different
    # hyperparameters and use the file './coefficient(figure3).py' to plot Fig.3(b)-(d).
    plot_figure2.plot_weights_path(args)
    plot_figure2.plot_loss_acc(args)
    plot_figure2.plot_weights_var_mean(args)

    # Plot Fig.4-5 in the paper
    # plot_figure4_5.plot_rescale_same(args)
    # plot_figure4_5.plot_rescale_different(args)

    # Plot Fig.6 in the paper
    # plot_figure6.plot_all_loss_acc(args)
    # plot_figure6.plot_weight_decay_path(args)
    # plot_figure6.plot_weight_decay_var_mean(args)
    # plot_figure6.plot_weight_decay_loss_acc(args)

    end_time = time.time()
    print('elapsed time is {} mins'.format((end_time-start_time)/60))


if __name__ == "__main__":
    main()
