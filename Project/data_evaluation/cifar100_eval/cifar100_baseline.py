from keras.datasets import cifar10, cifar100
import numpy as np
# classes_name_list=[i for i in range(10)]
import tensorflow as tf
from Project.data_evaluation.eval_class import eval_class
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model = tf.keras.models.load_model('CNN_with_dropout.h5')
# model = tf.keras.models.load_model('../models/cifar100_models/')

(X_train, y_train), (X_test, y_test) = cifar100.load_data()


"""
CNN_with_dropout
"""
model_name_list = [
    # "CNN_with_dropout.h5",
    # "CNN_without_dropout.h5",
    # "ResNet_v1.h5",
    # "ResNet_v2.h5",
    # "lenet5_with_dropout.h5",
    # "lenet5_without_dropout.h5",
    "random1_cifar100.h5",
    "random2_cifar100.h5"
]

for model_name_tmp in model_name_list:
    model_name=model_name_tmp
    dataset_name = "cifar-100"
    classes_num = 100
    batch_size = 20000
    # x_file = "../../Data/aug_imgs_cifar100/aug_imgs_cifar100_crop_x.npy"
    # y_file = "../../Data/aug_imgs_cifar100/aug_imgs_cifar100_crop_y.npy"
    x_true = X_train[:20000]
    y_true= y_train[:20000]
    augmentation_policy=""
    # model=tf.keras.models.load_model(model_name)
    model_pre = "../models/cifar100_models/"
    model = tf.keras.models.load_model(model_pre + model_name)
    eval_class_CNN_with_dropout = eval_class(
        dataset_name=dataset_name,
        classes_num=classes_num,
        batch_size=batch_size,
        x_true=x_true,
        y_true=y_true,
        augmentation_policy=augmentation_policy,
        model=model,
        model_name=model_name
    )

    accuracy1 = eval_class_CNN_with_dropout.predicting(0)
    print(model_name+"  "+augmentation_policy+"  "+str(accuracy1))
