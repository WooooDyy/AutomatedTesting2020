from keras.datasets import cifar10
import numpy as np
classes_name_list=[i for i in range(10)]

import tensorflow as tf
from Project.data_evaluation.eval_class import eval_class

model = tf.keras.models.load_model('CNN_with_dropout.h5')

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#
# test_img = skimage.io.imread(filename)  # 加载测试图像，直接得到多维数组
# test_img = skimage.transform.resize(test_img, (224, 224, 3))  # 根据模型输入尺寸要求，重定义大小
# test_img = np.expand_dims(test_img, 0)  # 模型输入要求为四维张量，所以需要增加一个batch_size维度
test_img = X_train[:10]
test_img = test_img/255
# test_img = np.expand_dims(test_img,0)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

pred = model.predict(test_img)  # 将测试图像送入模型，进行预测


def accuracy(y_hat, y):
    m = tf.argmax(y_hat, axis=1)
    return np.mean((tf.argmax(y_hat, axis=1) == y))

eval_class1 = eval_class(
    dataset_name="cifat-10",
    classes_num=10,
    batch_size=10,
    x_true=X_train,
    y_true=y_train,
    augmentation_policy=None,
    model=model
)
accuracy1 = eval_class1.predicting(0)
print(accuracy1)


y_hat = pred
yyyy = tf.argmax(y_hat, axis=1)
y_true = y_train[:10]
y_true =tf.reshape(y_true,shape=[-1])
y_true = tf.cast(y_true, dtype=tf.int64)

print(accuracy(pred,y_true))

print('预测结果：', end='')
print(classes_name_list[0])
print(pred.argmax())
print(str(classes_name_list[pred[0].argmax()]))  # 输出类别预测结果
print(y_train[0][0])


