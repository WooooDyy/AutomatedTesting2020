"""
属性：
    1. 数据集（cifar10或cifar100）
    2. 数据类别classes_list（10或100）
    3. 批量大小batch_size
    4. 增强数据集 x_true，y_true
    5. 增广策略（比如crop、shift等）
    6. 模型
函数：
    1. init（）：初始化数据集、数据类别list、批量大小、增强数据集
    2. accuracy（）：准确率，用来评估某个模型在某种增广策略下的表现
    3. predictiong（）：预测函数，用来预测某一张图片的分类，之后会调用accuracy（）来进行评估准确率
"""
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class eval_class:
    dataset_name = None
    classes_list = None
    batch_size = 0
    x_true = None
    y_true = None
    augmentation_policy = None
    model = None
    model_name = None
    def __init__(self,dataset_name,classes_num,batch_size,x_true,
                 y_true,augmentation_policy,model,model_name):
        self.dataset_name = dataset_name
        if(classes_num==10):
            self.classes_list = [i for i in range(10)]
        else:
            # todo
            self.classes_list = [i for i in range(100)]
        self.batch_size = batch_size
        self.x_true = x_true
        self.y_true = y_true
        self.augmentation_policy = augmentation_policy
        self.model = model
        self.model_name = model_name
    """
    计算批量的预测准确率
    """
    def accuracy(self,y_hat,y):
        return np.mean((tf.argmax(y_hat, axis=1) == y))
    """
    预测
    """
    def predicting(self,batch_idx):
        test_img = self.x_true[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        test_img = test_img / 255
        pred = self.model.predict(test_img)  # 将测试图像送入模型，进行预测
        y_hat = pred
        y_true_batch = self.y_true[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        y_true_batch = tf.reshape(y_true_batch, shape=[-1])
        y_true_batch = tf.cast(y_true_batch, dtype=tf.int64)
        return self.accuracy(y_hat=y_hat,y=y_true_batch)


