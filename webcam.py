import cv2
import numpy as np
import time
import os
import errno

import tensorflow as tf
#import torch
#from torch.nn import functional as F
#from torch.autograd import Variable as V
#from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
from collections import OrderedDict, deque

#from model import ConvColumn
#import torch.nn as nn
import json

import imutils
from imutils.video import VideoStream, FileVideoStream, WebcamVideoStream, FPS
import argparse
import pyautogui
import configparser

qsize = 20
sqsize = 8
num_classes = 8
threshold = 0.7

# from train_data.classes_dict in train.py
gesture_dict = {
    'Doing other things': 0, 0: 'Doing other things',
    'No gesture': 1, 1: 'No gesture', 
    'Stop Sign': 2, 2: 'Stop Sign', 
    'Swiping Left': 3, 3: 'Swiping Left', 
    'Swiping Right': 4, 4: 'Swiping Right', 
    'Swiping Up': 5, 5: 'Swiping Up', 
    'Turning Hand Clockwise': 6, 6: 'Turning Hand Clockwise', 
    'Turning Hand Counterclockwise': 7, 7: 'Turning Hand Counterclockwise'
}

# construct the argument parse and parse the arguments
str2bool = lambda x: (str(x).lower() == 'true')
parser = argparse.ArgumentParser()
# parser.add_argument('model')nppnpp
parser.add_argument("-e", "--execute", type=str2bool, default=True, help="Bool indicating whether to map output to keyboard/mouse commands")
parser.add_argument("-d", "--debug", type=str2bool, default=True, help="In debug mode, show webcam input")
parser.add_argument("-v", "--video", default='test.mp4', help="Path to video file if using an offline file")
parser.add_argument("-vb", "--verbose", default=2, help="Verbosity mode. 0- Silent. 1- Print info messages. 2- Print info and debug messages")
parser.add_argument("-cp", "--checkpoint", default="./model_best.h5", help="Location of model checkpoint file")
parser.add_argument("-m", "--mapping", default="./mapping.ini", help="Location of mapping file for gestures to commands")
args = parser.parse_args()

parser.print_help()
verbose = args.verbose

# read in configuration file for mapping of gestures to keyboard keys
mapping = configparser.ConfigParser()
action = {}
if os.path.isfile(args.mapping):
    mapping.read(args.mapping)
    for m in mapping['MAPPING']: 
        val = mapping['MAPPING'][m].split(',')
        action[m] = {'fn': val[0], 'keys': val[1:]}  # fn: hotkey, press, typewrite

else:
    # print('[ERROR] Mapping file for gestures to keyboard keys is not found at ' + args.mapping)
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.mapping)

if os.path.isfile(args.checkpoint):
    model = tf.keras.models.load_model(args.checkpoint)
    
else:
    # print("[ERROR] No checkpoint found at '{}'".format(args.checkpoint))
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.checkpoint)


# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
if verbose>0: print("[INFO] Attemping to start video stream...")

if (args.video == ''):
    vs = VideoStream(0, usePiCamera=False).start()
else:
    vs = FileVideoStream(args.video).start()

time.sleep(2.0)
fps = FPS().start()
Q = deque(maxlen=qsize)
SQ = deque(maxlen=sqsize)
act = deque(['No gesture', "No gesture"], maxlen=3)

# get first frame and use it to initialize our deque
frame = vs.read()
if frame is None:
    print('[ERROR] No video stream is available')

else:
    for i in range(qsize):
        Q.append(frame)
    if (verbose > 0): print('[INFO] Video stream started...')


# loop over the frames from the video stream
while(True):
    # grab the frame from the threaded video stream 
    frame = vs.read()
    if frame is None: 
        print('[ERROR] No video stream is available')
        break

    oframe = cv2.flip(frame.copy(), 1)  # copy original frame for display later as mirror image
    
    Q.append(frame)

    imgs = []
    for img in Q:
        img = tf.image.resize(img, (100, 100))/255
        imgs.append(img)
    
    pred = model.predict(np.expand_dims(imgs, axis=0))
    k = 5
    kth = pred[0].argpartition(-k)[::-1][:k]
    val = pred[0][kth]
    top5 = [gesture_dict[kth[i]] for i in range(k)]

    pi = [kth[i] for i in range(k)]
    ps = [val[i] for i in range(k)]
    top1 = top5[0] if ps[0] > threshold else gesture_dict[0]

    hist = {}
    for i in range(num_classes):
        hist[i] = 0
    for i in range(len(pi)):
        hist[pi[i]] = ps[i]
    SQ.append(list(hist.values()))

    ave_pred = np.array(SQ).mean(axis=0)
    top1 = gesture_dict[np.argmax(ave_pred)] if max(ave_pred) > threshold else gesture_dict[0]

    # show the output frame
    if (args.debug):
        cv2.putText(oframe, top1 + ' %.2f' % ps[0], (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(oframe, top1 + ' %.2f' % ps[0], (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.imshow("Frame", oframe)

    top1 = top1.lower()
    act.append(top1)

    # control an application based on mapped outputs
    # same top1 for consecutive frames
    if (act[0] != act[1] and len(set(list(act)[1:])) == 1):
        if top1 in action.keys():            
            t = action[top1]['fn']
            k = action[top1]['keys']
            
            if verbose > 1: print('[DEBUG]', top1, '-- ', t, str(k))
            if t == 'typewrite':
                pyautogui.typewrite(k)
            elif t == 'press':
                pyautogui.press(k)
            elif t == 'hotkey':
                for key in k:
                    pyautogui.keyDown(key)
                for key in k[::-1]:
                    pyautogui.keyUp(key)
                # pyautogui.hotkey(",".join(k))
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()