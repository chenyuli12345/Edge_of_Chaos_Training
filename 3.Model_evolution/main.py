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

# Training settings，设置训练参数

#创建一个argparse.ArgumentParser对象，用于保存所有需要的信息，一边命令行参数解析到python数据类型
parser = argparse.ArgumentParser(description='Asymptotic stability study of SGD') #提供一个描述信息

#添加一些命令行参数，每个参数都有一个名称、类型、默认值和描述信息
#数据集，默认为'mnist'
parser.add_argument('--dataset', type=str, default='mnist', help="datset {'mnist', 'kmnist', 'emnist/mnist'}. default: 'mnist'") #--dataset参数，类型为字符串，默认值为mnist，描述信息为'datset {'mnist', 'kmnist', 'emnist/mnist'}. default: 'mnist'
#激活函数，默认为'relu'
parser.add_argument('--activation-func', type=str, default='relu', help='activation function for hidden layers') #--activation-func参数，类型为字符串，默认值为'relu'，描述信息为'activation function for hidden layers'
#训练轮数，默认为4
parser.add_argument('--epochs', default=4, type=int, metavar='N', help='number of total epochs to run, should > 3') #--epochs参数，类型为整数，默认值为4，描述信息为'number of total epochs to run, should > 3'
#批次大小，默认为32
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='batch size for training') #--batch-size参数，类型为整数，默认值为32，描述信息为'batch size for training';metavar='N'是在帮助信息中显示的参数值的名称, 如果在命令行中输入--help，那么会看到类似于--batch-size N这样的信息。
#优化器，默认为'SGD'
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer used for training') #--optimizer参数，类型为字符串，默认值为'SGD'，描述信息为'optimizer used for training'
#学习率，默认为0.01
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr') #--lr参数，类型为浮点数，默认值为0.01，描述信息为'initial learning rate'，dest='lr'表示将参数保存到args.lr中（可以通过此代码来获取这个值）
#SGD算法的动量，默认为0.9
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for SGD') #--momentum参数，类型为浮点数，默认值为0.9，描述信息为'momentum for SGD'
#Adam算法的beta_1，默认为0.9
parser.add_argument('--beta-1', default=0.9, type=float, metavar='M', help='beta_1 in Adam') #--beta-1参数，类型为浮点数，默认值为0.9，描述信息为'beta_1 in Adam'
#Adam算法的beta_2，默认为0.999
parser.add_argument('--beta-2', default=0.999, type=float, metavar='M', help='beta_2 in Adam') #--beta-2参数，类型为浮点数，默认值为0.999，描述信息为'beta_2 in Adam'
#权重衰减，默认为0
parser.add_argument('--weight-decay', default=0, type=float, metavar='W', help='weight decay (default: 0)', dest='weight_decay') #--weight-decay参数，类型为浮点数，默认值为0，描述信息为'weight decay (default: 0)'，dest='weight_decay'表示将参数保存到args.weight_decay中
#重复次数，默认为10
parser.add_argument('--num-repeats', type=int, default=10, help='number of simulation repeats') #--num-repeats参数，类型为整数，默认值为10，描述信息为'number of simulation repeats'


def main():
    start_time = time.time() #开始记录当前时间
    args = parser.parse_args() #解析命令行参数，将参数保存到args中
    args.log_epochs = np.arange(args.epochs) #创建一个数组，从0到args.epochs-1
    #这是一个字符串，将所有参数组合在一起，用于后续的文件名，分别为“数据集_优化器_激活函数/epoch数量_10000倍学习率_批量大小_1000倍SGD算法动量_1000倍Adam算法参数beta1_10000000倍Adam算法参数beta2_1000000倍权重衰减”
    args.arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
        args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(args.weight_decay*1000000))
    #三个新的字符串，用于后续的文件名，最后一个参数均为上面的字符串args.arguments
    args.save_weights_dir = 'rawdata/weights/{}'.format(args.arguments)
    args.save_losses_dir = 'rawdata/losses/{}'.format(args.arguments)
    args.save_scores_dir = 'rawdata/scores/{}'.format(args.arguments)

    # (x_train, y_train), (x_test, y_test) = utils.load_qmnist_data()
    (x_train, y_train), (x_test, y_test) = utils.load_data(args.dataset) #加载数据集，x代表数据，y代表标签
    args.input_shape = x_train.shape[1] #获取数据的第二维度

    for num_repeat in range(args.num_repeats): #循环args.num_repeats次
        # break
        print('num_repeat={}'.format(num_repeat)) #打印当前循环次数
        args.save_weights_dir = 'rawdata/weights/{}/{}'.format(args.arguments, num_repeat) #更新代表文件名的变量，加上当前循环次数
        args.save_losses_dir = 'rawdata/losses/{}/{}'.format(args.arguments, num_repeat)
        args.save_scores_dir = 'rawdata/scores/{}/{}'.format(args.arguments, num_repeat)


        #设置神经网络模型
        if args.activation_func == 'relu': #如果激活函数为'relu'
            activation_func = tf.nn.relu #设置激活函数为relu
            model = tf.keras.Sequential([ #创立一个顺序模型，这是最简单的神经网络模型，由多个网络层线性堆叠而成
                tf.keras.layers.Dense(args.input_shape, activation=activation_func, name='layer_1',
                                      use_bias=False, input_shape=(args.input_shape,),
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay),), #全连接层，输入维度为args.input_shape，激活函数为activation_func（relu），不使用偏置，输入形状为(args.input_shape,)，使用L2正则化，args.weight_decay为正则化系数
                tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay)) #全连接层，输出维度为10，激活函数为softmax，使用L2正则化，args.weight_decay为正则化系数
            ])
        elif args.activation_func == 'tanh': #如果激活函数为'tanh'
            activation_func = tf.nn.tanh #设置激活函数为tanh
            model = tf.keras.Sequential([ #创立一个顺序模型，这是最简单的神经网络模型，由多个网络层线性堆叠而成
                tf.keras.layers.Dense(args.input_shape, activation=activation_func, name='layer_1',
                                      kernel_initializer=tf.keras.initializers.RandomNormal(
                                          mean=(0)/(args.input_shape), stddev=1/(2*np.sqrt(args.input_shape))),
                                      use_bias=False, input_shape=(args.input_shape,),
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay),), #全连接层，输入维度为args.input_shape，激活函数为activation_func（tanh），使用正态分布初始化，不使用偏置，输入形状为(args.input_shape,)，使用L2正则化，args.weight_decay为正则化系数
                tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      # kernel_initializer=tf.keras.initializers.RandomNormal(
                                      #     mean=(0)/(args.input_shape), stddev=1/(2*np.sqrt(args.input_shape))),
                                      kernel_regularizer=keras.regularizers.l2(args.weight_decay)) #全连接层，输出维度为10，激活函数为softmax，使用L2正则化，args.weight_decay为正则化系数
            ])

        #设置优化器
        if args.optimizer == 'SGD': #如果优化器为'SGD'
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum, nesterov=False, name='SGD') #设置优化器为SGD，学习率为args.lr，动量为args.momentum，不使用Nesterov动量
        elif args.optimizer == 'Adam': #如果优化器为'Adam'
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=1e-07, amsgrad=False, name='Adam') 
            #设置优化器为Adam，学习率为args.lr，beta_1为args.beta_1，beta_2为args.beta_2，epsilon（一个非常小的数，用于防止除以0的错误）为1e-07，这里没有使用AMSGrad版本的Adam
        
        
        metric_loss = SparseCategoricalCrossentropy(from_logits=False, name='sparse_categorical_crossentropy') #创建一个SparseCategoricalCrossentropy对象，这是一个损失函数，用于多分类问题，第一个参数表示模型的输出已经通过softmax函数转换为概率分布
        #编译模型，指定优化器、损失函数和评估指标
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[metric_loss, 'accuracy']) #优化器为上一步设置的optimizer，损失函数为sparse_categorical_crossentropy，评估指标为sparse_categorical_crossentropy和accuracy
        
        #创建一个LoggingReporter对象，用于记录训练过程中的信息
        reporter = loggingreporter.LoggingReporter(args, x_train, y_train, x_test, y_test)
        # model.fit(x_train, y_train, epochs=args.epochs,
        #           verbose=0, callbacks=[reporter, ], validation_split=0.2)
        
        #训练模型，前两个参数为训练数据，第三个参数为训练轮数，第四个参数为批次大小，verbose表示是否在标准输出流打印日志信息，callbacks表示在训练过程中调用的回调函数，validation_data表示验证数据
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0, callbacks=[reporter, ], validation_data=(x_test, y_test))

        #清理TensorFlow的后端绘画，释放资源
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
