from keras.datasets import cifar100
import numpy as np
# classes_name_list=[i for i in range(10)]
import tensorflow as tf
from Project.data_evaluation.eval_class import eval_class
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model = tf.keras.models.load_model('CNN_with_dropout.h5')
# model = tf.keras.models.load_model('../models/cifar100_models/')

# (X_train, y_train), (X_test, y_test) = cifar100.load_data()


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

base = {
    "CNN_with_dropout.h5":0.44,
    "CNN_without_dropout.h5":0.82,
    "ResNet_v1.h5":0.63,
    "ResNet_v2.h5":0.74,
    "lenet5_with_dropout.h5":0.55,
    "lenet5_without_dropout.h5":0.9,
    "random1_cifar100.h5":0.5,
    "random2_cifar100.h5":0.35
}

import csv
with open("../../../Data/cifar100_tables/cifar100_accuracy.csv", "w") as csvfile1,\
     open("../../../Data/cifar100_tables/cifar100_accuracy_minus.csv", "w") as csvfile2,\
        open("../../../Data/cifar100_tables/cifar100_accuracy_loss_rate.csv", "w") as csvfile3:
    writer1 = csv.writer(csvfile1)
    writer2 = csv.writer(csvfile2)
    writer3 = csv.writer(csvfile3)
    writer1.writerow(["",
                     "crop",
                    "shift",
                    "rotate",
                    "fliplr",
                    "flipud",
                    "additive_Gaussian_noise",
                    "brightness",
                    "contrast",
                    "crop_rotate_brightness",
                    "shift_noise"])
    writer2.writerow(["",
                      "crop",
                      "shift",
                      "rotate",
                      "fliplr",
                      "flipud",
                      "additive_Gaussian_noise",
                      "brightness",
                      "contrast",
                      "crop_rotate_brightness",
                      "shift_noise"])
    writer3.writerow(["",
                      "crop",
                      "shift",
                      "rotate",
                      "fliplr",
                      "flipud",
                      "additive_Gaussian_noise",
                      "brightness",
                      "contrast",
                      "crop_rotate_brightness",
                      "shift_noise"])

    for model_name in model_name_list:
        acc_list = [model_name]
        acc_minus_list = [model_name]
        acc_loss_rate_list = [model_name]
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
            acc_list = acc_list+[accuracy1]
            acc_minus_list = acc_minus_list+[accuracy1-base[model_name]]
            acc_loss_rate_list = acc_loss_rate_list+[(accuracy1-base[model_name])/base[model_name]]
            print("cifar-100"+"     "+model_name+"       "+augmentation_policy+"       "+str(accuracy1))
        writer1.writerow(acc_list)
        writer2.writerow(acc_minus_list)
        writer3.writerow(acc_loss_rate_list)
#todo 跑完eval cifar10的，填进去，在搞cifar100的