import os
import glob

import numpy as np
from math import ceil
from random import random, randint
from PIL import Image
from skimage import color
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


##################################################################################
# Training Data Processing
##################################################################################

class SISRData:

    def __init__(self, args):
        self.factor = args.factor
        self.bicubic_size = args.bicubic_size
        self.gaussian_size = args.gaussian_size
        self.anti_aliasing = args.anti_aliasing
        self.channels = args.channels
        self.training_data_path = args.training_data_path
        self.validation_data_path = args.validation_data_path
        self.test_data_path = args.test_data_path
        self.test_label_path = args.test_label_path
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.Qsize = args.Qsize

        path_train = os.path.join(self.training_data_path, '*')
        path_val = os.path.join(self.validation_data_path, '*')
        path_test_lr = os.path.join(self.test_data_path, '*')
        path_test_hr = os.path.join(self.test_label_path, '*')
        print("###### path_train ", path_train)
        print("###### path_val ", path_val)
        print("###### path_test_lr ", path_test_lr)
        print("###### path_test_hr ", path_test_hr)
        self.list_train = sorted(glob.glob(path_train))
        self.list_val = sorted(glob.glob(path_val))
        self.list_test_lr = sorted(glob.glob(path_test_lr))
        self.list_test_hr = sorted(glob.glob(path_test_hr))
        self.num_train = len(self.list_train)

        print('Load all files list')
        print("# training imgs : {} \n".format(self.num_train))

        # bicubic kernel to be convolved by anisotropic gaussian
        self.bicubic_kernel = get_bicubic_kernel(self.bicubic_size, anti_aliasing=self.anti_aliasing, factor=self.factor)
        self.bicubic_kernel = tf.constant(self.bicubic_kernel, dtype=tf.float32, shape=(1, self.bicubic_size, self.bicubic_size, 1))
        self.bicubic_kernel = tf.tile(self.bicubic_kernel, [1, 1, 1, self.batch_size])
        self.pad_left = (self.bicubic_size - self.factor) // 2
        self.pad_right = self.pad_left

    def image_processing(self, img_path):
        y, gaussian_kernel = tf.py_func(self.image_processing_py, [img_path], [tf.float32, tf.float32])
        y.set_shape((self.Qsize, self.factor * self.patch_size, self.factor * self.patch_size, self.channels))
        gaussian_kernel.set_shape((self.Qsize, self.gaussian_size, self.gaussian_size, 1))
        return y, gaussian_kernel

    def image_processing_py(self, img_path):
        img_hr = Image.open(img_path)

        width, height = img_hr.size
        patches_hr = np.zeros((self.Qsize, self.factor * self.patch_size, self.factor * self.patch_size, self.channels), dtype=np.float32)
        patches_gaussian_kernel = np.zeros((self.Qsize, self.gaussian_size, self.gaussian_size), dtype=np.float32)
        for patch in range(self.Qsize):
            w = int(random() * (width - self.patch_size * self.factor))
            h = int(random() * (height - self.patch_size * self.factor))
            patches_hr[patch] = np.array(
                img_hr.crop((w, h, w + self.patch_size * self.factor, h + self.patch_size * self.factor)), 'float32')
            patches_gaussian_kernel[patch] = random_anisotropic_gaussian_kernel(width=self.gaussian_size)

        if random() > 0.5:  # horizontal flip
            patches_hr = np.flip(patches_hr, axis=2)

        rot = randint(0, 3)
        patches_hr = np.rot90(patches_hr, rot, (1, 2))

        patches_hr = (patches_hr / 255.0) * 2 - 1  # normalize to [-1,1]
        patches_gaussian_kernel = np.expand_dims(patches_gaussian_kernel, -1)

        return patches_hr, patches_gaussian_kernel


##################################################################################
# Degradation
##################################################################################

def get_bicubic_kernel(bicubic_size, anti_aliasing=False, factor=1):
    # set correct factor if anti_aliasing=True
    # assert self.bicubic_size % 2 == 0, "bicubic_size should be an even number"
    cubic_input = np.arange(-bicubic_size // 2 + 1, bicubic_size // 2 + 1) - 0.5
    if anti_aliasing:
        bicubic_kernel = cubic32(cubic_input / float(factor))
    else:
        bicubic_kernel = cubic32(cubic_input)
    bicubic_kernel = bicubic_kernel / np.sum(bicubic_kernel)
    bicubic_kernel = np.outer(bicubic_kernel, bicubic_kernel.T)
    return bicubic_kernel


def cubic32(x):
    x = np.array(x).astype(np.float32)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def inv_covariance_matrix(sig_x, sig_y, theta):
    # sig_x : x-direction standard deviation
    # sig_x : y-direction standard deviation
    # theta : rotation angle
    D_inv = np.array([[1/(sig_x ** 2), 0.], [0., 1/(sig_y ** 2)]])  # inverse of diagonal matrix D
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # eigenvector matrix
    inv_cov = np.dot(U, np.dot(D_inv, U.T))  # inverse of covariance matrix
    return inv_cov


def anisotropic_gaussian_kernel(width, inv_cov):
    # width : kernel size of anisotropic gaussian filter
    ax = np.arange(-width // 2 + 1., width // 2 + 1.)
    # avoid shift
    if width % 2 == 0:
        ax = ax - 0.5
    xx, yy = np.meshgrid(ax, ax)
    xy = np.stack([xx, yy], axis=2)
    # pdf of bivariate gaussian distribution with the covariance matrix
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inv_cov) * xy, 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def random_anisotropic_gaussian_kernel(width=15, sig_min=0.2, sig_max=4.0):
    # width : kernel size of anisotropic gaussian filter
    # sig_min : minimum of standard deviation
    # sig_max : maximum of standard deviation
    sig_x = np.random.random() * (sig_max - sig_min) + sig_min
    sig_y = np.random.random() * (sig_max - sig_min) + sig_min
    theta = np.random.random() * 3.141/2.
    inv_cov = inv_covariance_matrix(sig_x, sig_y, theta)
    kernel = anisotropic_gaussian_kernel(width, inv_cov)
    kernel = kernel.astype(np.float32)
    return kernel


def random_anisotropic_gaussian_kernel_seed(s, width=15, sig_min=0.2, sig_max=4.0):
    # width : kernel size of anisotropic gaussian filter
    # sig_min : minimum of standard deviation
    # sig_max : maximum of standard deviation
    # s as seed
    np.random.seed(3 * s)
    sig_x = np.random.random() * (sig_max - sig_min) + sig_min
    np.random.seed(3 * s + 1)
    sig_y = np.random.random() * (sig_max - sig_min) + sig_min
    np.random.seed(3 * s + 2)
    theta = np.random.random() * 3.141/2.
    inv_cov = inv_covariance_matrix(sig_x, sig_y, theta)
    kernel = anisotropic_gaussian_kernel(width, inv_cov)
    kernel = kernel.astype(np.float32)
    return kernel


##################################################################################
# Image I/O
##################################################################################

def read_img_trim(img_path, factor):
    # read and trim image so that it is divisible by factor
    img = np.array(Image.open(img_path), 'float32')
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = h - np.remainder(h, factor)
        w = w - np.remainder(w, factor)
        img = np.expand_dims(img[:h, :w, :], axis=0)
    else:
        h, w = img.shape
        h = h - np.remainder(h, factor)
        w = w - np.remainder(w, factor)
        img = np.expand_dims(img[:h, :w], axis=0)
    img = (img / 255.0) * 2.0 - 1.0
    return img


def save_img(img, img_path):
    img = np.squeeze(img)
    img = np.clip((img + 1.) / 2. * 255., 0, 255).round()
    img = Image.fromarray(img.astype('uint8'))
    img.save(img_path)


##################################################################################
# Image Processing
##################################################################################

def get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW):
    # get boundary indices for patch-wise processing
    H_low_ind = max(pH * sH - patch_boundary, 0)
    H_high_ind = min((pH + 1) * sH + patch_boundary, h)
    W_low_ind = max(pW * sW - patch_boundary, 0)
    W_high_ind = min((pW + 1) * sW + patch_boundary, w)

    return H_low_ind, H_high_ind, W_low_ind, W_high_ind


def trim_patch_boundary(img, patch_boundary, h, w, pH, sH, pW, sW, sf):
    # trim boundaries for patch-wise processing
    if patch_boundary == 0:
        img = img
    else:
        if pH * sH < patch_boundary:
            img = img
        else:
            img = img[:, patch_boundary*sf:, :, :]
        if (pH + 1) * sH + patch_boundary > h:
            img = img
        else:
            img = img[:, :-patch_boundary*sf, :, :]
        if pW * sW < patch_boundary:
            img = img
        else:
            img = img[:, :, patch_boundary*sf:, :]
        if (pW + 1) * sW + patch_boundary > w:
            img = img
        else:
            img = img[:, :, :-patch_boundary*sf, :]

    return img


##################################################################################
# Resize functions from https://github.com/fatheral/matlab_imresize
##################################################################################

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    # compute weights and indices from kernel function
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresize(I, scalar_scale=None, output_shape=None):
    kernel = cubic
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = imresizevec(B, weights[dim], indices[dim], dim)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)


##################################################################################
# Miscellaneous
##################################################################################

def compute_psnr(img_gt, img_out, peak):
    mse = np.mean(np.square(img_gt - img_out))
    psnr = 10 * np.log10(peak*peak / mse)
    return psnr


def compute_y_psnr(img_gt_rgb, img_out_rgb):
    # images must be in range [-1, 1] float or double
    peak = 255
    img_gt_rgb = np.squeeze(img_gt_rgb)
    img_out_rgb = np.squeeze(img_out_rgb)
    img_gt_rgb = np.clip((img_gt_rgb + 1.) / 2. * 255., 0, 255).round()
    img_out_rgb = np.clip((img_out_rgb + 1.) / 2. * 255., 0, 255).round()

    img_gt_yuv = color.rgb2ycbcr(img_gt_rgb.astype('uint8'))
    img_out_yuv = color.rgb2ycbcr(img_out_rgb.astype('uint8'))
    img_gt_yuv = np.clip(img_gt_yuv[:, :, 0], 0, 255).round()
    img_out_yuv = np.clip(img_out_yuv[:, :, 0], 0, 255).round()
    psnr = compute_psnr(img_gt_yuv, img_out_yuv, peak)
    return psnr


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def check_gray(img):
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=3)
        img = np.tile(img, (1, 1, 1, 3))
    return img
