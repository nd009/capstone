## Sys
import os
import io
import sys
import re
import errno
import shutil
import datetime
import time
import string
import copy
import subprocess as sp
from operator import itemgetter
from fractions import Fraction
## Data structure
import random
import json
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import six
## CV
import cv2
## Video visualization
import psycopg2
import psycopg2.extras
import matplotlib.pyplot as plt
import PIL
from pprint import pprint
import multiprocessing
## Model
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
## Parameters
import params ## you can modify the content of params

#########################################
#### Video control
#########################################
## Basic function
def is_sequence(arg):
    return (not hasattr(arg, 'strip') and 
            hasattr(arg, '__getitem__') and
            hasattr(arg, '__iter__'))

def is_int(s):
    assert not is_sequence(s)
    try: 
        int(s)
        return True
    except ValueError:
        return False

## Return data path for loader
def join_dir(dirpath, filename):
    return os.path.join(dirpath, filename)    

## Load csv file
def fetch_csv_data(path):
    return pd.read_csv(path)

## Load video file
def imread(img_path, mode=cv2.IMREAD_COLOR):
    assert os.path.isfile(img_path), 'Bad image path: {}'.format(img_path)
    return cv2.imread(img_path, mode)

def cv2_current_frame(cap):
    x = cap.get(cv2.CAP_PROP_POS_FRAMES)
    assert x.is_integer()
    return int(x)

def cv2_goto_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    assert cv2_current_frame(cap) == frame_id

##
def frame_count(path, method='ffmpeg'):
    if method == 'ffmpeg':
        return ffmpeg_frame_count(path)
    else:
        assert False 

##     
def ffmpeg_frame_count(path):
    cmd = 'ffmpeg -i {} -vcodec copy -acodec copy -f null /dev/null 2>&1'.format(path)
    cmd_res = sp.check_output(cmd, shell=True)
    cmd_res = copy.deepcopy(cmd_res)

    fc = None

    lines = cmd_res.splitlines()
    lines = lines[::-1]

    for line in lines:
        line = line.strip()
        # res = re.match(r'frame=\s*(\d+)\s*fps=', line)
        res = re.match(b'frame=\s*(\d+)\s*fps=', line)
        if res:
            fc = res.group(1)
            
            assert is_int(fc)
            fc = int(fc)
            break

    assert fc is not None

    return fc
##
def without_ext(path): 
    return os.path.splitext(path)[0]

def ext(path, period=False):
    x = os.path.splitext(path)[1]
    x = x.replace('.', '')
    return x

def mkv_to_mp4(mkv_path, remove_mkv=False):
    assert os.path.isfile(mkv_path)
    assert ext(mkv_path) == 'mkv'
    mp4_path = without_ext(mkv_path) + '.mp4'
    
    if os.path.isfile(mp4_path):
        os.remove(mp4_path)
    
    cmd = 'ffmpeg -i {} -c:v copy -c:a libfdk_aac -b:a 128k {} >/dev/null 2>&1'.format(mkv_path, mp4_path)
    sp.call(cmd, shell=True)

    assert os.path.isfile(mp4_path) # make sure that the file got generated successfully

    if remove_mkv:
        assert os.path.isfile(mkv_path)
        os.remove(mkv_path)
##
def video_resolution_to_size(resolution, width_first=True):
    if resolution == '720p':
        video_size = (1280, 720)
    elif resolution == '1080p':
        video_size = (1920, 1080)
    elif resolution == '1440p':
        video_size = (2560, 1440)
    elif resolution == '4k':
        video_size = (3840, 2160)
    else: assert False

    if not width_first:
        video_size = (video_size[1], video_size[0])
    return video_size
    
def cv2_resize_by_height(img, height):
    ratio = height / img.shape[0]
    width = ratio * img.shape[1]
    height, width = int(round(height)), int(round(width))
    return cv2.resize(img, (width, height))
              
## Video output
def overlay_image(l_img, s_img, x_offset, y_offset):
    assert y_offset + s_img.shape[0] <= l_img.shape[0]
    assert x_offset + s_img.shape[1] <= l_img.shape[1]

    l_img = l_img.copy()
    for c in range(0, 3):
        l_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1], c] = (
                  s_img[:,:,c] * (s_img[:,:,3]/255.0) +
                  l_img[y_offset:y_offset+s_img.shape[0],
                        x_offset:x_offset+s_img.shape[1], c] *
                  (1.0 - s_img[:,:,3]/255.0))
    return l_img


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape)/2)[:2]
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    return result


#########################################
#### Visualize result
#########################################
def get_human_steering(epoch_id):
    epoch_dir = params.data_dir
    assert os.path.isdir(epoch_dir)
    steering_path = join_dir(epoch_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
    assert os.path.isfile(steering_path)
    
    rows = fetch_csv_data(steering_path)
    human_steering = list(rows.wheel.values)
    return human_steering

def visualize(epoch_id, machine_steering, out_dir, perform_smoothing=False,
              verbose=False, verbose_progress_step = 100, frame_count_limit = None):
    epoch_dir = params.data_dir
    human_steering = get_human_steering(epoch_id)
    assert len(human_steering) == len(machine_steering)

    # testing: artificially magnify steering to test steering wheel visualization
    # human_steering = list(np.array(human_steering) * 10)
    # machine_steering = list(np.array(machine_steering) * 10)

    # testing: artificially alter machine steering to test that the disagreement coloring is working
    # delta = 0
    # for i in xrange(len(machine_steering)):
    #     delta += random.uniform(-1, 1)
    #     machine_steering[i] += delta
    
    if perform_smoothing:
        machine_steering = list(smooth(np.array(machine_steering)))

    steering_min = min(np.min(human_steering), np.min(machine_steering))
    steering_max = max(np.max(human_steering), np.max(machine_steering))

    assert os.path.isdir(epoch_dir)

    front_vid_path = join_dir(epoch_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(front_vid_path)
    
    dash_vid_path = join_dir(epoch_dir, 'epoch{:0>2}_dash.mkv'.format(epoch_id))
    dash_exists = os.path.isfile(dash_vid_path)

    front_cap = cv2.VideoCapture(front_vid_path)
    dash_cap = cv2.VideoCapture(dash_vid_path) if dash_exists else None
    
    assert os.path.isdir(out_dir)
    vid_size = video_resolution_to_size('720p', width_first=True)
    out_path = join_dir(out_dir, 'epoch{:0>2}_human_machine.mkv'.format(epoch_id))
    # vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'X264' ), 30, vid_size)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('X','2','6','4'), 30, vid_size)
    # vw = cv2.VideoWriter(out_path, cv2.cv.FOURCC(*'X264' ), 30, vid_size)
    w, h = vid_size

    # for f_cur in xrange(len(machine_steering)):
    for f_cur in range(len(machine_steering)):
        if (f_cur != 0) and (f_cur % verbose_progress_step == 0):
            print('completed {} of {} frames'.format(f_cur, len(machine_steering)))

        if (frame_count_limit is not None) and (f_cur >= frame_count_limit):
            break
            
        rret, rimg = front_cap.read()
        assert rret

        if dash_exists:
            dret, dimg = dash_cap.read()
            assert dret
        else:
            dimg = rimg.copy()
            dimg[:] = (0, 0, 0)
        
        ry0, rh = 80, 500
        dimg = dimg[100:, :930]
        dimg = cv2_resize_by_height(dimg, h-rh)

        fimg = rimg.copy()
        fimg[:] = (0, 0, 0)
        fimg[:rh] = rimg[ry0:ry0+rh]
        dh, dw = dimg.shape[:2]
        fimg[rh:,:dw] = dimg[:]
        
        ########################## plot ##########################
        plot_size = (500, dh)
        win_before, win_after = 150, 150

        xx, hh, mm = [], [], []
        # for f_rel in xrange(-win_before, win_after+1):
        for f_rel in range(-win_before, win_after+1):
            f_abs = f_cur + f_rel
            if f_abs < 0 or f_abs >= len(machine_steering):
                continue
            xx.append(f_rel/30)
            hh.append(human_steering[f_abs])
            mm.append(machine_steering[f_abs])

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        steering_range = max(abs(steering_min), abs(steering_max))
        #ylim = [steering_min, steering_max]
        ylim = [-steering_range, steering_range]
        # ylim[0] = min(np.min(hh), np.min(mm))
        # ylim[1] = max(np.max(hh), np.max(mm))
        
        axis.set_xlabel('Current Time (secs)')
        axis.set_ylabel('Steering Angle')
        axis.axvline(x=0, color='k', ls='dashed')
        axis.plot(xx, hh)
        axis.plot(xx, mm)
        axis.set_xlim([-win_before/30, win_after/30])
        axis.set_ylim(ylim)
        #axis.set_ylabel(y_label, fontsize=18)
        axis.label_outer()
        #axes.append(axis)

        buf = io.BytesIO()
        # http://stackoverflow.com/a/4306340/627517
        sx, sy = plot_size
        sx, sy = round(sx / 100, 1), round(sy / 100, 1)

        fig.set_size_inches(sx, sy)
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        buf_img = PIL.Image.open(buf)
        pimg = np.asarray(buf_img)
        plt.close(fig)

        pimg = cv2.resize(pimg, plot_size)
        pimg = pimg[:,:,:3]

        ph, pw = pimg.shape[:2]
        pimg = 255 - pimg
        
        fimg[rh:,-pw:] = pimg[:]

        ####################### human steering wheels ######################
        wimg = imread(os.path.abspath("images/wheel-tesla-image-150.png"), cv2.IMREAD_UNCHANGED)

        human_wimg = rotate_image(wimg, -human_steering[f_cur])
        wh, ww = human_wimg.shape[:2]
        fimg = overlay_image(fimg, human_wimg, y_offset = rh+50, x_offset = dw+60)

        ####################### machine steering wheels ######################
        disagreement = abs(machine_steering[f_cur] - human_steering[f_cur])
        machine_wimg = rotate_image(wimg, -machine_steering[f_cur])
        red_machine_wimg = machine_wimg.copy()
        green_machine_wimg = machine_wimg.copy()
        red_machine_wimg[:,:,2] = 255
        green_machine_wimg[:,:,1] = 255
        #r = disagreement / (steering_max - steering_min)
        max_disagreement = 10
        r = min(1., disagreement / max_disagreement)
        g = 1 - r
        assert r >= 0
        assert g <= 1
        machine_wimg = cv2.addWeighted(red_machine_wimg, r, green_machine_wimg, g, 0)
        wh, ww = machine_wimg.shape[:2]
        fimg = overlay_image(fimg, machine_wimg, y_offset = rh+50, x_offset = dw+260)
        
        ####################### text ######################
        timg_green_agree = imread(os.path.abspath("images/text-green-agree.png"), cv2.IMREAD_UNCHANGED)
        timg_ground_truth = imread(os.path.abspath("images/text-ground-truth.png"), cv2.IMREAD_UNCHANGED)
        timg_learned_control = imread(os.path.abspath("images/text-learned-control.png"), cv2.IMREAD_UNCHANGED)
        timg_red_disagree = imread(os.path.abspath("images/text-red-disagree.png"), cv2.IMREAD_UNCHANGED)
        timg_tesla_control_autopilot = imread(os.path.abspath("images/text-tesla-control-autopilot.png"), cv2.IMREAD_UNCHANGED)
        timg_tesla_control_human = imread(os.path.abspath("images/text-tesla-control-human.png"), cv2.IMREAD_UNCHANGED)

        fimg = overlay_image(fimg, timg_tesla_control_autopilot, y_offset = rh+8, x_offset = dw+83)
        fimg = overlay_image(fimg, timg_learned_control, y_offset = rh+8, x_offset = dw+256)
        fimg = overlay_image(fimg, timg_ground_truth, y_offset = rh+205, x_offset = dw+90)
        fimg = overlay_image(fimg, timg_red_disagree, y_offset = rh+205, x_offset = dw+230)
        fimg = overlay_image(fimg, timg_green_agree, y_offset = rh+205, x_offset = dw+345)

        if (frame_count_limit is not None) and (frame_count_limit == 1):
            cv2.imwrite(out_path.replace('mkv', 'jpg'), fimg)
            sys.exit()

        vw.write(fimg)

    front_cap.release()
    if dash_exists:
        dash_cap.release()
    vw.release()

    mkv_to_mp4(out_path, remove_mkv=True)



#########################################
#### Model load
#########################################
def save_model(model, epoch=''):
    """
    Saves the model and the weights to a json file
    :param model: The mode to be saved
    :param epoch: The epoch number, so as to save the model to a different file name after each epoch
    :return: None
    """
    model_path = join_dir(params.model_dir, 'model_{}.json'.format(epoch))
    param_path = join_dir(params.model_dir, 'model_{}.h5'.format(epoch))
    #
    json_string = model.to_json()

    with open(model_path, 'w') as outfile:
        outfile.write(json_string)
    model.save_weights(param_path)
    print('Model saved')
    
def get_model():
    """
    Defines the model
    :return: Returns the model
    """
    """
    Check if a model already exists
    """
    model_path = join_dir(params.model_dir, 'model.json')
    param_path = join_dir(params.model_dir, 'model.h5')
    
    if os.path.exists(model_path):
        ch = input('Model already exists, do you want to reuse? (y/n): ')
        if ch == 'y' or ch == 'Y':
            with open(model_path, 'r') as in_file:
                json_model = in_file.read()
                model = model_from_json(json_model)

            weights_file = os.path.join(param_path)
            model.load_weights(weights_file)
            print('Model fetched from the disk')
            model.summary()
    return model


#########################################
#### Testing pipeline
#########################################
if __name__ == '__main__':
    epoch_id = 1
    machine_steering = get_human_steering(epoch_id)

    # frame_count_limit = None
    # frame_count_limit = 30 * 5
    # frame_count_limit = 1
    visualize(epoch_id, machine_steering, params.out_dir,
              verbose=True, frame_count_limit=150)
