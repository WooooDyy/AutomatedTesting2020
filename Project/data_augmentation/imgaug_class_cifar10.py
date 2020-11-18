from keras.datasets import cifar10
from keras.datasets import cifar
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA

from imgaug import augmenters as iaa
import imgaug as ia

"""
数据扩增类，可以通过函数传入batch_idx，获取一系列的扩增后的图像。
"""

class imgaug_cifar10:
    x_train, y_train, x_test, y_test = None, None, None, None
    batch_size = 0

    def __init__(self, x_train, y_train,batch_size):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

    """
    剪切函数
    可以的，只要剪切是不要剪切太过，使得数据失真即可
    """

    def crop(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Crop(percent=(0, 0.25))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    """
    平移函数
    """

    def shift(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)})
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    """
    旋转函数
    不是mnist数字之类的数据集，因此旋转理论上来说应该是不会影响判断结果的。本文不把旋转度数加的太大。
    """

    def rotate(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-30, 30))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    """
    水平翻动
    """

    def fliplr(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5)
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    """
    垂直翻动
    """

    def flipud(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Flipud(0.5)
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    # """
    # 错切
    # """
    #
    # def shearing(self, batch_idx):
    #     seq = iaa.Sequential([
    #         iaa.Affine(shear=(-20, 20))
    #     ])
    #     images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
    #     images_aug = images_aug = seq.augment_images(images)
    #     return images_aug

    """
    明亮度
    """

    def brightness(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Multiply(mul=(0.8, 1.2))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    """
    对比度
    """

    def contrast(self, batch_idx):
        seq = iaa.Sequential([
            iaa.ContrastNormalization((0.75, 1.25))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug
    """
    噪点
    """

    def additive_Gaussian_noise(self, batch_idx):
        seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.2 * 255))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    # """
    # 畸变
    # """
    #
    # def salt_and_pepper(self, batch_idx):
    #     seq = iaa.Sequential([
    #         iaa.PiecewiseAffine(scale=(0.01, 0.07))
    #     ])
    #     images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
    #     images_aug = images_aug = seq.augment_images(images)
    #     return images_aug

    """
    裁剪+旋转+明亮度
    """
    def crop_rotate_brightness(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Multiply(mul=(0.9, 1.1)),
            iaa.Crop(percent=(0, 0.05)),
            iaa.Affine(rotate=(-10, 10))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug

    """
    平移+噪点插入
    """

    def shift_noise(self, batch_idx):
        seq = iaa.Sequential([
            iaa.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}),
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255))
        ])
        images = self.x_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images_aug = seq.augment_images(images)
        y_aug = self.y_train[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return images_aug,y_aug