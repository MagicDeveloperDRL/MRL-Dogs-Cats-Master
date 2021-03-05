'''''''''
@file: DataHelper.py
@author: MRL Liu
@time: 2021/3/3 15:27
@env: Python,Numpy
@desc:本模块为数据读取模块，
      （1）提供了读取一系列图片文件并进行预处理到DataSets对象的功能
      （2) 支持将训练数据划分为训练集和验证集。
      （2) 提供预处理图片方法、可视化读取后的图片文件数据的方法。
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random


"""图片数据集对象"""
class DataSet(object):
    def __init__(self,images,labels,img_names,cls):
        self._num_examples = images.shape[0] # 样本总数量

        self._images = images # 图片数据
        self._labels = labels # 图片标签（one-hot编码）
        self._img_names = img_names # 图片名称
        self._cls = cls # 图片标签语义
        self._epochs_done = 0 # 所有样本被取完一遍的次数
        self._index_in_epoch = 0 # 上一批次取后的序号
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def img_names(self):
        return self._img_names
    @property
    def cls(self):
        return self._cls
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self,batch_size):
        assert batch_size <= self._num_examples # 检测批次大小是否超过样本总数

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:# 如果取完后的该批次大于样本总数
            self._epochs_done += 1 # 取完次数+1
            start = 0 # 从头开始数
            self._index_in_epoch = batch_size #

        end = self._index_in_epoch

        return self._images[start:end],self._labels[start:end],self._img_names[start:end],self.cls[start:end]
"""图片全部数据集对象"""
class DataSets(object):
    def __init__(self, train=None, valid=None,test = None):
        self.__train = train  # 训练数据集
        self.__valid = valid  # 验证数据集
        self.__test = test # 测试数据集

    @property
    def train(self):
        return self.__train

    @property
    def valid(self):
        return self.__valid

    @property
    def test(self):
        return self.__test

    def set_train(self,train):
        self.__train = train

    def set_valid(self,valid):
        self.__valid = valid

    def set_test(self,test):
        self.__test = test

"""图片数据读取辅助类"""
class DataHelper(object):
    def __init__(self, train_path, classes,image_size=64,test_path=None):
        """
        :param train_path: 训练图片数据路径
        :param classes: 标签的类别，字符串数组
        :param image_size: 图片缩放后的尺寸，默认为正方形
        :param test_path: 测试图片数据路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.image_size = image_size
        self.classes = classes


    def load_data(self,data_path):
        """
        :param data_path: 读取的图片数据的路径
        :return: 返回图片集、标签集、文件名集、标签语义集
        """
        images = []  # 图片集
        labels = []  # 图片标签集
        img_names = []  # 图片文件名集
        cls = []  # 图片标签语义集
        if data_path == None:
            print('无法读取{}文件夹下的图片！'.format(data_path))
            return
        else:
            print('正在读取{}文件夹下的图片...'.format(data_path))
        # 遍历存储数据
        for fields in self.classes:
            index = self.classes.index(fields)  # 获取该分类的序号
            print('现在计划读取{} 文件夹(Index: {})'.format(fields, index))
            path = os.path.join(data_path, fields, '*g')  # 图片文件的格式
            files = glob.glob(path)  # 返回符合规则的文件路径列表
            for f1 in files:
                # 读取图片
                image =preprocessed_image(f1,self.image_size)
                #image = cv2.imread(filename=f1)  # 从路径下读取彩色图片
                #image = cv2.resize(image, dsize=(self.image_size, self.image_size), fx=0, fy=0,
                   #                interpolation=cv2.INTER_LINEAR)  # 图片缩放大小
                #image = image.astype(np.float32)  # 转换图片数组的数据类型
                #image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                # 设置标签
                label = np.zeros(len(self.classes))
                label[index] = 1.0
                labels.append(label)
                # 存储图片文件名和标签语义
                flbase = os.path.basename(f1)  # 返回path最后的文件名
                img_names.append(flbase)
                cls.append(fields)
        # 转换数据格式
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        cls = np.array(cls)

        return images, labels, img_names, cls


    def get_data_sets(self,validation_size,is_read_test=True):
        """
        :param validation_size: 验证集在训练数据中的比例，0-1之间的小数，如0.2
        :param is_read_test: 是否读取测试数据集，注意需要初始化该类时设置测试数据的路径
        :return: DataSets对象
        """
        data_sets = DataSets()
        # 读取训练数据集并将其拆分成训练集和验证集
        images, labels, img_names, cls = self.load_data(self.train_path)
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls) # python函数，随机排序

        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_cls = cls[validation_size:]

        data_sets.set_train(DataSet(train_images,train_labels,train_img_names,train_cls))
        data_sets.set_valid(DataSet(validation_images,validation_labels,validation_img_names,validation_cls))
        print("成功读取Training-set中的文件数量:\t{}".format(len(data_sets.train.labels)))
        print("成功读取Validation-set中的文件数量:\t{}".format(len(data_sets.valid.labels)))
        # 读取测试数据集
        if is_read_test and self.test_path!=None:
            test_images, test_labels, test_img_names, test_cls = self.load_data(self.test_path)
            test_images, test_labels, test_img_names, test_cls = shuffle(test_images, test_labels, test_img_names, test_cls)  # python函数，随机排序
            data_sets.set_test(DataSet(test_images, test_labels, test_img_names, test_cls))
            print("成功读取Testing-set中的文件数量:\t{}".format(len(data_sets.test.labels)))
        return data_sets

def plot_images(images, cls_true, img_size=64, cls_pred=None, num_channels=3):
    # 检测图像是否存在
    if len(images) == 0:
        print("没有图像来展示")
        return
    # 随机采样9张图像
    random_indices = random.sample(range(len(images)), min(len(images), 9))
    images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

    # 创造一个3行3列的画布
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    fig.canvas.set_window_title('Random Images show')  # 设置字体大小与格式
    for i, ax in enumerate(axes.flat):
        # 显示图片
        if len(images) < i + 1:
            break
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # 展示图像的语义标签和实际预测标签
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # 设置每张图的标签为其xlabel.
        ax.set_xlabel(xlabel)

        # 设置图片刻度
        ax.set_xticks([0, img_size])
        ax.set_yticks([0, img_size])

    plt.show()

def preprocessed_image(image_path, image_size, is_imshow=False):
    image = cv2.imread(image_path)  # 读取图片,返回一个三维数组：（x，y,(R,G,B)）shape:(333,500,3)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)  # 图像缩放大小 shape:(64,64,3)
    image = image.astype('float32')  # 转为浮点数
    image = np.multiply(image, 1.0 / 255.0)  # 转为小数
    image = np.array(image)
    # 是否显示转换后的数据
    if is_imshow:
        plt.imshow(image)
        # print(image.shape)
    return image


if __name__=='__main__':
    # 获取数据集
    dataHelper = DataHelper(train_path='training_data',
                            test_path='testing_data',
                            classes = ['dogs', 'cats'],
                            image_size=64)
    data = dataHelper.get_data_sets(validation_size=0.2,
                                    is_read_test=False)
    # 获取训练的一些数据并且进行显示
    images, cls_true = data.train.images, data.train.cls
    plot_images(images=images, cls_true=cls_true)

