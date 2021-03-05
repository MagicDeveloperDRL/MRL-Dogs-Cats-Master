'''''''''
@file: Model_Constructor.py
@author: MRL Liu
@time: 2021/3/3 13:25
@env: Python,Numpy,TensorFlow,OpenCV-Python,matplotlib,scikit-learn
@desc:本模块为模型构造器，负责构建和训练模型
      （1）支持基于TensorFlow搭建CNN模型
      （2) 支持基于TensorFlow训练CNN模型，支持早期终止机制、定期保存模型数据和训练日志。
      （3）支持可视化训练过程中的模型的损失值、精度等变化。
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import json
import os
import time
from datetime import timedelta
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import DataHelper
plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

Net_Parameter_Save_Path = "config/train_data.json" # 网络参数保存路径
Net_Parameter_Load_Path = "config/train_data.json" #

"""模型构造器"""
class Model_Constructor(object):
    def __init__(self,classes,img_size,num_channels):
        self.num_channels = num_channels
        self.img_size = img_size
        self.classes = classes
        self.num_classes = len(self.classes)
        self.log_path ="logs" # log日志，用于TensorBoard
        # 训练监测数据列表
        self.train_acc_list = [] # 训练精度列表
        self.val_acc_list = [] # 验证精度列表
        self.train_loss_list = []  # 训练精度列表
        self.val_loss_list = [] # 验证精度列表
        self.iter_list = [] # 迭代次数列表

    def train_model(self,num_iteration,data,batch_size):
        total_iterations = 0
        saver = tf.train.Saver()
        # 早期终止机制所需的变量
        early_stopping = None
        best_val_loss = float("inf")
        patience = 0
        # 记录开始训练的时刻
        start_time = time.time()
        # 开始训练
        for i in range(total_iterations, total_iterations + num_iteration):

            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

            feed_dict_tr = {self.x: x_batch, self.y_true: y_true_batch}
            feed_dict_val = {self.x: x_valid_batch, self.y_true: y_valid_batch}
            # 执行优化器操作
            self.session.run(self.optimizer, feed_dict=feed_dict_tr)
            # 定期检测损失
            if i % int(data.train.num_examples / batch_size) == 0:
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                tr_loss = self.session.run(self.cost, feed_dict=feed_dict_tr)
                epoch = int(i / int(data.train.num_examples / batch_size))

                self.__show_pregress(epoch, feed_dict_tr, feed_dict_val, val_loss, tr_loss,i)
                saver.save(self.session, './dogs-cats-model/dog-cat.ckpt', global_step=i)
                # 早期终止机制
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience == early_stopping:
                        break
        # 计算训练花费的时间并打印
        total_iterations += num_iteration
        end_time = time.time()
        time_dif = end_time - start_time
        print("本次训练总共花费的Time: " + str(timedelta(seconds=int(round(time_dif)))))
        # 输出训练信息到日志中
        writer = tf.summary.FileWriter(self.log_path, self.session.graph)
        writer.close()
        path = os.path.dirname(os.path.abspath(__file__))+"\\"+self.log_path
        print("训练日志已经保存至路径：{} ".format(path))


    def __show_pregress(self,epoch, feed_dict_train, feed_dict_validata, val_loss,tr_loss, i):
        tra_acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validata)
        msg = "训练回合数:{0}-迭代次数:{1},训练精度:{2:>6.1%},验证精度:{3:>6.1%},验证损失值:{4:.3f}"
        print(msg.format(epoch + 1, i, tra_acc, val_acc, val_loss))

        self.train_acc_list.append(tra_acc*100)  # 训练精度列表
        self.val_acc_list.append(val_acc*100)  # 验证精度列表
        self.train_loss_list.append(tr_loss)  # 验证精度列表
        self.val_loss_list.append(val_loss)  # 验证精度列表
        self.iter_list.append(epoch)  # 迭代次数列表


    def plot_training_loss_and_accuracy(self):
        # 创建画布
        fig = plt.figure(figsize=(12, 6))  # 创建一个指定大小的画布
        fig.canvas.set_window_title('猫狗识别器的训练情况')
        # 添加第1个窗口
        ax1 = fig.add_subplot(121)  # 添加一个1行2列的序号为1的窗口
        # 添加标注
        ax1.set_title('训练中的损失变化', fontsize=14)  # 设置标题
        ax1.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax1.set_ylabel('y(损失大小)', fontsize=14, fontstyle='oblique')
        ax1.set_ylim(0, 1)
        ax1.xaxis.set_major_locator(MultipleLocator(1))  # 设置x轴的间隔
        # 绘制函数
        line1, = ax1.plot(self.iter_list, self.train_loss_list, color='blue', label="训练集")
        line2, = ax1.plot(self.iter_list, self.val_loss_list, color='red', label="验证集")
        ax1.legend(handles=[line1, line2], loc=2)  # 绘制图例说明
        # plt.grid(True)#启用表格
        # 添加第2个窗口
        ax2 = fig.add_subplot(122)  # 添加一个1行2列的序号为1的窗口
        # 添加标注
        ax2.set_title('训练中的正确率变化', fontsize=14)  # 设置标题
        ax2.set_xlabel('x(训练次数)', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax2.set_ylabel('y(正确率%)', fontsize=14, fontstyle='oblique')
        ax2.set_ylim(0, 100)
        ax2.xaxis.set_major_locator(MultipleLocator(1))# 设置x轴的间隔
        # 绘制函数
        line1, = ax2.plot(self.iter_list, self.train_acc_list, color='blue', label="训练集")
        # 绘制函数
        line2, = ax2.plot(self.iter_list, self.val_acc_list, color='red', label="验证集")
        ax2.legend(handles=[line1, line2], loc=1)  # 绘制图例说明

        # plt.grid(True) #启用表格


    def save_training_loss_and_accuracy(self, filename):
        """
        # 保存训练检测数据
        :param filename: '../config/record.json'
        :return:
        """
        data = {"train_acc_list": [w.tolist() for w in self.train_acc_list],
                "val_acc_list": [w.tolist() for w in self.val_acc_list],
                "train_loss_list": [w.tolist() for w in self.val_acc_list],
                "val_loss_list": [w.tolist() for w in self.val_loss_list],
                "iter_list": [w for w in self.iter_list]}
        f = open(filename, "w")
        json.dump(data, f)  # 将Python数据结构编码为JSON格式并且保存至文件中
        f.close()  # 关闭文件
        print("训练检测数据成功保存至{}文件".format(filename))


    def load_training_loss_and_accuracy(self,filename):
        """
        # 读取训练检测数据
        """
        f = open(filename, "r")
        data = json.load(f)  # 将文件中的JSON格式解码为Python数据结构
        f.close()
        self.train_acc_list = [np.array(w) for w in data["train_acc_list"]]
        self.val_acc_list = [np.array(w) for w in data["val_acc_list"]]
        self.train_loss_list = [np.array(w) for w in data["train_loss_list"]]
        self.val_loss_list = [np.array(w) for w in data["val_loss_list"]]
        self.iter_list = [ w for w in data["iter_list"]]
        print("训练检测数据已成功读取到构造器中...")
        return

    def create_model(self,learning_rate=1e-4):
        # 定义输入数据结构
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        y_true_cls = tf.argmax(self.y_true, axis=1)
        # 创建CNN模型，返回输出
        layer_fc2 = self.__create_cnn(self.x)
        y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
        y_pred_cls = tf.argmax(y_pred, axis=1, name='y_pred_cls')
        # 定义损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=self.y_true)
        self.cost = tf.reduce_mean(cross_entropy)
        # 选择优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        # 定义检测数据
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 初始化所有变量
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def __create_cnn(self,x_image):
        # 第1层卷积
        filter_size_conv1 = 3
        num_filters_conv1 = 32
        # 第2层卷积
        filter_size_conv2 = 3
        num_filters_conv2 = 32
        # 第3层卷积
        filter_size_conv3 = 3
        num_filters_conv3 = 64
        # 第5、6层全连接
        fc_layer_size = 1024
        print('输入层', x_image.get_shape())  #(?, 64, 64, 3)
        layer_conv1 = self.__create_convolutional_layer(input=x_image,
                                                      num_input_channels=self.num_channels,
                                                      conv_filter_size=filter_size_conv1,
                                                      num_filters=num_filters_conv1)
        print('layer_conv1', layer_conv1.get_shape())  #(?, 32, 32, 32)
        layer_conv2 = self.__create_convolutional_layer(input=layer_conv1,
                                                      num_input_channels=num_filters_conv1,
                                                      conv_filter_size=filter_size_conv2,
                                                      num_filters=num_filters_conv2)
        print('layer_conv2', layer_conv2.get_shape())  #(?, 16, 16, 32)
        layer_conv3 = self.__create_convolutional_layer(input=layer_conv2,
                                                      num_input_channels=num_filters_conv2,
                                                      conv_filter_size=filter_size_conv3,
                                                      num_filters=num_filters_conv3)

        print('layer_conv3', layer_conv3.get_shape())  #(?, 8, 8, 64)
        layer_flat = self.__create_flatten_layer(layer_conv3)
        print('layer_flat', layer_flat.get_shape())  #(?, 4096)
        layer_fc1 = self.__create_fc_layer(input=layer_flat,
                                         num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                         num_outputs=fc_layer_size,
                                         use_relu=True)
        print('layer_fc1', layer_fc1.get_shape())  #(?, 1024)
        layer_fc2 = self.__create_fc_layer(input=layer_fc1,
                                         num_inputs=fc_layer_size,
                                         num_outputs=self.num_classes,
                                         use_relu=False)
        print('layer_fc2', layer_fc2.get_shape())  #(?, 2)

        return layer_fc2

    def __create_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def __create_biases(self,size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    def __create_convolutional_layer(self,input,
                                   num_input_channels,
                                   conv_filter_size,
                                   num_filters):
        weights = self.__create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = self.__create_biases(num_filters)

        layer = tf.nn.conv2d(input=input, # 输入的原始张量
                             filter=weights, # 卷积核张量，(filter_height、 filter_width、in_channels,out_channels)
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += biases
        layer = tf.nn.relu(layer)

        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        return layer

    def __create_flatten_layer(self,layer):
        layer_shape = layer.get_shape()
        print('扁平前',layer_shape)#扁平前 (?, 8, 8, 64)
        num_features = layer_shape[1:4].num_elements()
        print('[1:4]:', layer_shape[1:4]) #[1:4]: (8, 8, 64)
        print('num_features:', num_features)#num_features: 4096
        re_layer = tf.reshape(layer, [-1, num_features])
        print('扁平后', re_layer.get_shape()) #扁平后 (?, 4096)
        return re_layer

    def __create_fc_layer(self,input,
                        num_inputs,
                        num_outputs,
                        use_relu=True):
        weights = self.__create_weights(shape=[num_inputs, num_outputs])
        biases = self.__create_biases(num_outputs)
        layer = tf.matmul(input, weights) + biases
        layer = tf.nn.dropout(layer, keep_prob=0.7)
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer




if __name__=='__main__':
    image_size = 64
    num_channels = 3
    classes = ['dogs', 'cats']
    use_old_train_data = True
    constructor = Model_Constructor(classes=classes,img_size=image_size,num_channels=num_channels)
    if use_old_train_data:
        # 获取数据集
        dataHelper = DataHelper.DataHelper(train_path='training_data',  # 训练数据存储路径
                                           test_path='testing_data',
                                           classes=classes,
                                           image_size=image_size)
        data = dataHelper.get_data_sets(validation_size=0.2,  # 验证集的比例
                                        is_read_test=False)
        # 创建模型
        constructor.create_model()
        # 训练模型
        constructor.train_model(num_iteration = 100,# 迭代次数
                      data =data, # 训练数据
                      batch_size = 32 # 批大小
                      )
        # 保存训练过程中的训练监控数据
        constructor.save_training_loss_and_accuracy(Net_Parameter_Save_Path)
    else:
        constructor.load_training_loss_and_accuracy(Net_Parameter_Save_Path)
    # 展示训练过程中的训练监控数据
    constructor.plot_training_loss_and_accuracy()