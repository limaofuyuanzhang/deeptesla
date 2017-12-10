# coding:utf-8
import cv2
import pandas as pd
import numpy as np
import params
import random
import h5py
import os
from keras.utils import HDF5Matrix
import sys


# 图片处理
def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 50px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0] / 3):shape[0] - 50, 0:shape[1]]
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    return img

def frame_count_func(file_path):
    '''return frame count of this video'''
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count


def load_train():
    epochs = [1, 2, 3, 4, 5, 6, 7, 8]
    imgs = []
    wheels = []
    # extract image and steering data
    for epoch in epochs:

        img_path = os.path.join(
            './epochs', 'epoch{:0>2}_front.mkv'.format(epoch))
        frame_count = frame_count_func(img_path)
        cap = cv2.VideoCapture(img_path)

        csv_path = os.path.join(
            './epochs', 'epoch{:0>2}_steering.csv'.format(epoch))
        rows = pd.read_csv(csv_path)
        wheels.extend(rows['wheel'].values)

        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = img_pre_process(img)
            imgs.append(img)
        cap.release()

    augmented_imgs = []
    augmented_wheels = []
    shuffle_augmented_imgs = []
    shuffle_augmented_wheels = []
    for image, wheel in zip(imgs, wheels):
        augmented_imgs.append(image)
        augmented_wheels.append(wheel)
        # 翻转图片
        flipped_image = cv2.flip(image, 1)
        flipped_wheel = float(wheel) * -1.0
        augmented_imgs.append(flipped_image)
        augmented_wheels.append(flipped_wheel)
        #打乱顺序
    index = [i for i in range(len(augmented_imgs))]
    random.shuffle(index)
    for i in range(len(augmented_imgs)):     
        shuffle_augmented_imgs.append(augmented_imgs[index[i]])
        shuffle_augmented_wheels.append(augmented_wheels[index[i]])
                      
    X_train = np.array(shuffle_augmented_imgs)
    y_train = np.array(shuffle_augmented_wheels)
    y_train = np.reshape(y_train,(len(y_train),1))
    return X_train, y_train

def load_data(num):
    imgs = []
    wheels = []
    img_path = os.path.join(
            './epochs', 'epoch{:0>2}_front.mkv'.format(num))
    frame_count = frame_count_func(img_path)
    cap = cv2.VideoCapture(img_path)

    csv_path = os.path.join(
            './epochs', 'epoch{:0>2}_steering.csv'.format(num))
    rows = pd.read_csv(csv_path)
    wheels.extend(rows['wheel'].values)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = img_pre_process(img)
        imgs.append(img)
    cap.release()
    
    X_train = np.array(imgs)
    y_train = np.array(wheels)
    y_train = np.reshape(y_train,(len(y_train),1))
    return X_train, y_train
    
    
    
def load_data1(mode, color_mode='RGB', flip=True):
    '''get train and valid data,
    mode: train or valid, color_mode:RGB or YUV
    output: batch data.'''
    if mode == 'train':
        epochs = [1, 2, 3, 4, 5, 6, 7, 8]
    elif mode == 'valid':
        epochs = [9]
    elif mode == 'test':
        epochs = [10]
    else:
        print('Wrong mode input')

    imgs = []
    wheels = []
    # extract image and steering data
    for epoch_id in epochs:
        yy = []

        vid_path = os.path.join(
            './epochs', 'epoch{:0>2}_front.mkv'.format(epoch_id))
        frame_count = frame_count_func(vid_path)
        cap = cv2.VideoCapture(vid_path)

        csv_path = os.path.join(
            './epochs', 'epoch{:0>2}_steering.csv'.format(epoch_id))
        rows = pd.read_csv(csv_path)
        yy = rows['wheel'].values
        wheels.extend(yy)

        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = img_pre_process(img)
            imgs.append(img)

        assert len(imgs) == len(wheels)

        cap.release()
        
    if mode == 'train' and flip:
        augmented_imgs = []
        augmented_measurements = []
        shuffle_augmented_imgs = []
        shuffle_augmented_measurements = []
        for image, measurement in zip(imgs, wheels):
            augmented_imgs.append(image)
            augmented_measurements.append(measurement)
            # Flip images
            flipped_image = cv2.flip(image, 1)
            flipped_measurement = float(measurement) * -1.0
            augmented_imgs.append(flipped_image)
            augmented_measurements.append(flipped_measurement)
          #打乱顺序
        index = [i for i in range(len(augmented_imgs))]
        random.shuffle(index)
        for i in range(len(augmented_imgs)):     
            shuffle_augmented_imgs.append(augmented_imgs[index[i]])
            shuffle_augmented_measurements.append(augmented_measurements[index[i]])
                      
        X_train = np.array(shuffle_augmented_imgs)
        y_train = np.array(shuffle_augmented_measurements)
        y_train = np.reshape(y_train,(len(y_train),1))
    else:
        # 如果是test或者是不翻转则直接使用
        X_train = np.array(imgs)
        y_train = np.array(wheels)
        y_train = np.reshape(y_train,(len(y_train),1))

    return X_train, y_train


# 移除数据
def remove_data():
    h5_path = './epochs/deep_tesla_origin.hdf5'
    if os.path.exists(h5_path):
        os.remove(h5_path)


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

