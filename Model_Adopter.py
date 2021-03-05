'''''''''
@file: Model_Adopter.py
@author: MRL Liu
@time: 2021/3/3 15:27
@env: Python,Numpy,TensorFlow,OpenCV-Python,matplotlib,scikit-learn
@desc:本模块为模型采用器，负责调用训练好的模型
      （1）支持基于TensorFlow搭建CNN模型
      （2) 支持基于TensorFlow训练CNN模型，支持早期终止机制、定期保存模型数据和训练日志。
      （3）支持可视化训练过程中的模型的损失值、精度等变化。
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''

import DataHelper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support



class Model_Adopter(object):
    def __init__(self):
        # 加载训练好的模型
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-9975.meta')
        saver.restore(self.sess, './dogs-cats-model/dog-cat.ckpt-9975')
        graph = tf.get_default_graph()
        # 重建变量
        self.x = graph.get_tensor_by_name("x:0")
        self.y_true = graph.get_tensor_by_name("y_true:0")
        self.y_pred = graph.get_tensor_by_name("y_pred:0")
        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)


    def sample_prediction(self,images):
        # 运行模型得出结果
        feed_dict_test = {
            self.x: images,
            self.y_true: np.zeros((len(images),2))
        }
        result = self.sess.run(self.y_pred, feed_dict=feed_dict_test)
        res_label = ['dogs', 'cats']
        result = [res_label[r.argmax()] for r in result]
        return result


    def plot_example_errors(self,data,cls_pred, correct):
        # 获取不正确的数组
        incorrect = (correct == False)
        # 裁剪数组
        num = len(incorrect)
        images = data.images[:num]
        cls_pred = cls_pred[:num]
        cls_true = data.cls[:num]
        # 显示图像
        images = images[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = cls_true[incorrect]
        DataHelper.plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

    def plot_confusion_matrix(self,data,cls_pred,num_classes=2):
        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.
        cls_true = data.cls

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

        # Compute the precision, recall and f1 score of the classification
        p, r, f, s = precision_recall_fscore_support(cls_true, cls_pred, average='weighted')
        print('Precision:', p)
        print('Recall:', r)
        print('F1-score:', f)

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.matshow(cm)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def print_validation_accuracy(self,data,test_batch_size=20,show_example_errors=False, show_confusion_matrix=False):
        # 预测结果
        num_test = 0
        y_true_cls = []
        y_pred_cls = []
        # 遍历一次数据集
        while num_test<data.num_examples :
            # 获取批次数量
            images, y_true, _, _y_true_cls = data.next_batch(test_batch_size)
            num_test += len(_y_true_cls)
            if num_test>data.num_examples: # 如果要检测的样本数超过了测试总数
                break
            else:
                _y_pred_cls = model_Adopter.sample_prediction(images)
                y_true_cls.extend(_y_true_cls)
                y_pred_cls.extend(_y_pred_cls)

        # 计算精度
        y_true_cls = np.array(y_true_cls)
        y_pred_cls = np.array(y_pred_cls)
        correct = (y_true_cls == y_pred_cls)
        num_test = len(correct)
        correct_sum = correct.sum()
        acc = float(correct_sum) / num_test
        # 打印精度
        print("测试数据的精度: {0:.1%} ({1} / {2})".format(acc, correct_sum, num_test))

        # 显示一些预测错误的样本
        if show_example_errors:
            print("Example errors:")
            self.plot_example_errors(data,cls_pred=y_pred_cls, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            self.plot_confusion_matrix(data,cls_pred=y_pred_cls)


if __name__=="__main__":
    image_size = 64  # 图像尺寸
    num_channels = 3  # 通道数
    train_path = 'training_data'
    classes = ['dogs', 'cats']
    validation_size = 0.2
    img_size_flat = image_size * image_size * num_channels
    dog_path = 'Test_image/dog.jpg'  # 图片路径
    cat_path = 'Test_image/cat.jpg'
    # 创建模型应用器
    model_Adopter =  Model_Adopter()

    # 应用模型
    # 自己加载一些图片进行应用
    test_cat = DataHelper.preprocessed_image(cat_path,image_size=image_size)
    test_dog = DataHelper.preprocessed_image(dog_path,image_size=image_size)
    test_images = [test_cat, test_dog]
    test_images_label_pred = model_Adopter.sample_prediction(test_images)
    test_images_label = ['cat', 'dog']
    DataHelper.plot_images(test_images, test_images_label, cls_pred=test_images_label_pred)


    # 获取数据集
    dataHelper = DataHelper.DataHelper(train_path='training_data',
                            test_path='testing_data',
                            classes=['dogs', 'cats'],
                            image_size=64)
    data = dataHelper.get_data_sets(validation_size=0.2,
                                    is_read_test=True)
    model_Adopter.print_validation_accuracy(data=data.test,
                                            test_batch_size=20,
                                            show_example_errors=True,
                                            show_confusion_matrix=True)


