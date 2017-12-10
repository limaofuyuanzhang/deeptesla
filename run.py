#!/usr/bin/env python
# coding:utf-8

import sys
import os
import time
import subprocess as sp
import itertools
## CV
import cv2
## Model
import numpy as np
import tensorflow as tf
## Tools
import utils
## Parameters
import params ## you can modify the content of params.py

## Test epoch
epoch_ids = [10]
## Load model,获取模型
model = utils.get_model()

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


## Process video
for epoch_id in epoch_ids:
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(vid_path)
    frame_count = utils.frame_count(vid_path)
    cap = cv2.VideoCapture(vid_path)

    machine_steering = []

    print('performing inference...')
    time_start = time.time()
    for frame_id in range(frame_count):
        ret, img = cap.read()
        assert ret
        ## you can modify here based on your model
        img = img_pre_process(img)
        img = img[None,:,:,:]
        deg = float(model.predict(img, batch_size=1))
        machine_steering.append(deg)

    cap.release()

    fps = frame_count / (time.time() - time_start)
    
    print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))
    
    print('performing visualization...')
    utils.visualize(epoch_id, machine_steering, params.out_dir,
                        verbose=True, frame_count_limit=None)
    
    
