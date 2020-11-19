from keras.datasets import cifar100
from keras.datasets import cifar
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA

from imgaug import augmenters as iaa
import imgaug as ia

# from . import imgaug_class_cifar100
from Project.data_augmentation.imgaug_class_cifar100 import imgaug_cifar100


def show9(imgs):
    ''' Create a grid of 3x3 images
    '''
    if imgs.max() > 1:
        imgs = imgs / 255.

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(imgs[i], cmap=plt.get_cmap())
    plt.show()

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

imgaug_exec_c100 = imgaug_cifar100(x_train=X_train,y_train=y_train,batch_size=10000)


# imgs = imgaug_exec_c10.crop(0)
# np.save("test.npy",imgs)
# test = np.load("test.npy")
# show9(test)
"""
存储的时候，每一份都存储x和y，使用的时候分别读取
"""


"""
crop
"""
batch_idx = 0
aug_imgs_cifar100_crop_x,aug_imgs_cifar100_crop_y = imgaug_exec_c100.crop(batch_idx)
# np.save("test.npy",(aug_imgs_cifar10_crop_x,aug_imgs_cifar10_crop_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_crop_x.npy",aug_imgs_cifar100_crop_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_crop_y.npy",aug_imgs_cifar100_crop_y)
# batch_idx+=1


aug_imgs_cifar100_shift_x,aug_imgs_cifar100_shift_y = imgaug_exec_c100.shift(batch_idx)
# np.save("test.npy",(aug_imgs_cifar10_shift_x,aug_imgs_cifar10_shift_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_shift_x.npy",aug_imgs_cifar100_shift_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_shift_y.npy",aug_imgs_cifar100_shift_y)
# batch_idx+=1

aug_imgs_cifar100_rotate_x,aug_imgs_cifar100_rotate_y = imgaug_exec_c100.rotate(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_rotate_x,aug_imgs_cifar100_rotate_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_rotate_x.npy",aug_imgs_cifar100_rotate_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_rotate_y.npy",aug_imgs_cifar100_rotate_y)
# batch_idx+=1

aug_imgs_cifar100_fliplr_x,aug_imgs_cifar100_fliplr_y = imgaug_exec_c100.fliplr(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_fliplr_x,aug_imgs_cifar100_fliplr_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_fliplr_x.npy",aug_imgs_cifar100_fliplr_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_fliplr_y.npy",aug_imgs_cifar100_fliplr_y)
# batch_idx+=1

aug_imgs_cifar100_flipud_x,aug_imgs_cifar100_flipud_y = imgaug_exec_c100.flipud(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_flipud_x,aug_imgs_cifar100_flipud_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_flipud_x.npy",aug_imgs_cifar100_flipud_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_flipud_y.npy",aug_imgs_cifar100_flipud_y)
# batch_idx+=1

aug_imgs_cifar100_additive_Gaussian_noise_x,aug_imgs_cifar100_additive_Gaussian_noise_y = imgaug_exec_c100.additive_Gaussian_noise(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_additive_Gaussian_noise_x,aug_imgs_cifar100_additive_Gaussian_noise_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_additive_Gaussian_noise_x.npy",aug_imgs_cifar100_additive_Gaussian_noise_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_additive_Gaussian_noise_y.npy",aug_imgs_cifar100_additive_Gaussian_noise_y)
# batch_idx+=1

aug_imgs_cifar100_brightness_x,aug_imgs_cifar100_brightness_y = imgaug_exec_c100.brightness(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_brightness_x,aug_imgs_cifar100_brightness_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_brightness_x.npy",aug_imgs_cifar100_brightness_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_brightness_y.npy",aug_imgs_cifar100_brightness_y)
# batch_idx+=1

aug_imgs_cifar100_contrast_x,aug_imgs_cifar100_contrast_y = imgaug_exec_c100.contrast(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_contrast_x,aug_imgs_cifar100_contrast_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_contrast_x.npy",aug_imgs_cifar100_contrast_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_contrast_y.npy",aug_imgs_cifar100_contrast_y)
# batch_idx+=1

aug_imgs_cifar100_shift_noise_x,aug_imgs_cifar100_shift_noise_y = imgaug_exec_c100.shift_noise(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_shift_noise_x,aug_imgs_cifar100_shift_noise_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_shift_noise_x.npy",aug_imgs_cifar100_shift_noise_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_shift_noise_y.npy",aug_imgs_cifar100_shift_noise_y)
# batch_idx+=1

aug_imgs_cifar100_crop_rotate_brightness_x,aug_imgs_cifar100_crop_rotate_brightness_y = imgaug_exec_c100.crop_rotate_brightness(batch_idx)
# np.save("test.npy",(aug_imgs_cifar100_crop_rotate_brightness_x,aug_imgs_cifar100_crop_rotate_brightness_y))
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_crop_rotate_brightness_x.npy",aug_imgs_cifar100_crop_rotate_brightness_x)
np.save("../../Data/aug_imgs_cifar100/aug_imgs_cifar100_crop_rotate_brightness_y.npy",aug_imgs_cifar100_crop_rotate_brightness_y)
# batch_idx+=1




# aug_imgs_2_shift_x = imgaug_exec_c10.shift(batch_idx)
# np.save("aug_imgs_2_shift_batch.npy",aug_imgs_2_shift_x)
# 
# batch_idx+=1


