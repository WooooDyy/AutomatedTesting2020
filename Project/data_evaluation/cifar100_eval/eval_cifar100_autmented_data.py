from keras.datasets import cifar100
import numpy as np
# classes_name_list=[i for i in range(10)]
import tensorflow as tf
from Project.data_evaluation.eval_class import eval_class
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model = tf.keras.models.load_model('CNN_with_dropout.h5')
# model = tf.keras.models.load_model('../models/cifar100_models/')

(X_train, y_train), (X_test, y_test) = cifar100.load_data()


model_name_list = [
    "CNN_with_dropout.h5",
    "CNN_without_dropout.h5",
    "ResNet_v1.h5",
    "ResNet_v2.h5",
    "lenet5_with_dropout.h5",
    "lenet5_without_dropout.h5",
    "random1_cifar100.h5",
    "random2_cifar100.h5"
]
pre_string = "aug_imgs_cifar100_"
post_string_x="_x.npy"
post_string_y = "_y.npy"
aug_policy_list = [
    "crop",
    "shift",
    "rotate",
    "fliplr",
    "flipud",
    "additive_Gaussian_noise",
    "brightness",
    "contrast",
    "crop_rotate_brightness",
    "shift_noise"
]

for model_name in model_name_list:
    for policy in aug_policy_list:
        dataset_name = "cifar-100"
        classes_num = 100 # todo
        batch_size = 10000
        x_file = "../../../Data/aug_imgs_cifar100/"+pre_string+policy+post_string_x
        y_file = "../../../Data/aug_imgs_cifar100/"+pre_string+policy+post_string_y
        x_true = np.load(x_file)
        y_true= np.load(y_file)
        augmentation_policy=policy
        model_pre = "../models/cifar100_models/"
        model=tf.keras.models.load_model(model_pre+model_name)
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
        print("cifar-100"+"     "+model_name+"       "+augmentation_policy+"       "+str(accuracy1))

#todo 跑完eval cifar10的，填进去，在搞cifar100的