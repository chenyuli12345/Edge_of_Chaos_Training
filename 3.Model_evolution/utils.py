import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy import io as spio #导入scipy库的io模块，并将其重命名为spio。scipy是一个用于科学计算和技术计算的Python库，而io模块提供了一些输入/输出功能
from tensorflow import keras
import pickle #用来在文件之间存储和读取python对象，实现了基本的数据序列化（将对象信息保存到文件中）和反序列化（从文件中创建出原来的对象）



#函数用于加载和预处理tensorflow指定的数据集，接收参数dataset_name，最后返回的是两个元组，第一个元组包含训练数据和训练标签，第二个元组包含测试数据和测试标签
def load_data(dataset_name):
    #加载MNIST数据集，返回的是一个元组，元组中包含两个元组，第一个元组包含训练数据和训练标签，第二个元组包含测试数据和测试标签

    #tfds.load表示从Tensorflow数据集库中加载指定的数据集，第一个参数为数据集名称；第二个参数为数据集的划分方式（这里分为train和test）；第三个参数为批处理大小（这里表示加载所有数据）；第四个参数表示是否以监督方式加载数据，这里表示数据将以输入/输出对的形式加载
    #tfds.as_numpy表示将数据集转换为numpy数组
    (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(dataset_name, split=['train', 'test'], batch_size=-1, as_supervised=True,))
    
    #将x_train和x_test的形状调整为二维，第一维度保持不变

    #np.reshape表示将数据集的形状转换为指定的形状，第一个参数为数据集，第二个参数为新的形状；astype表示将数据集的数据类型转换为指定的数据类型
    x_train = np.reshape(x_train, [x_train.shape[0], -1]).astype('float32')/255. #这里表示第一维度保持不变，第二维度将所有剩余元素平铺开来，还要除以255进行归一化
    x_test = np.reshape(x_test, [x_test.shape[0], -1]).astype('float32')/255.

    return (x_train, y_train), (x_test, y_test)

#用于加载QMNIST（一种常用的手写数字识别数据集），返回的是两个元组，第一个元组包含训练数据和训练标签，第二个元组包含测试数据和测试标签
def load_qmnist_data():
    mnist = pickle.load(open('./data/qmnist.pkl', "rb")) #pickle.load表示从文件中加载数据，第一个参数为文件路径，第二个参数为读取模式（二进制模式），得到的是一个字典
    x_train, y_train = mnist['train_data'], mnist['train_labels'] #从字典中取出训练数据和训练标签，x表示训练数据，y表示训练标签
    x_test, y_test = mnist['test_data'], mnist['test_labels'] #从字典中取出测试数据和测试标签，x表示测试数据，y表示测试标签

    #np.reshape表示将数据集的形状转换为指定的形状，第一个参数为数据集，第二个参数为新的形状；astype表示将数据集的数据类型转换为指定的数据类型
    x_train = np.reshape(x_train, [x_train.shape[0], -1]).astype('float32')/255. #这里表示第一维度保持不变，第二维度将所有剩余元素平铺开来，还要除以255进行归一化
    x_test = np.reshape(x_test, [x_test.shape[0], -1]).astype('float32')/255. #这里表示第一维度保持不变，第二维度将所有剩余元素平铺开来，还要除以255进行归一化

    return (x_train, y_train), (x_test, y_test) #返回两个元组，第一个元组包含训练数据和训练标签，第二个元组包含测试数据和测试标签
