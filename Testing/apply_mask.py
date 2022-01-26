#apply_mask.py
from __future__ import print_function
import time
import cv2 as cv
import argparse
import os
import numpy as np
import imageio
from PIL import Image

main_path = os.path.dirname(os.path.abspath(__file__)) 
original_path = os.path.join(main_path, "preprocess")
mask_path = os.path.join(main_path, "postprocess")
original = os.path.join(original_path, "test2.mp4")
mask = os.path.join(mask_path, "test.mp4")
save_path = os.path.join(main_path, "postprocess2")

original_cap = cv.VideoCapture(original)
mask_cap = cv.VideoCapture(mask)
frames = []

def main():
    while True:
        
        oret, oframe = original_cap.read()
        mret, mframe = mask_cap.read()
        if oframe is None or mframe is None:
            break
    
        img_gray = cv.cvtColor(mframe, cv.COLOR_BGR2GRAY)
        frame = cv.bitwise_or(oframe, oframe, mask=img_gray)
        #cv.imshow('Frame', oframe)
        #cv.waitKey(1)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        start_recording(frame, frames)

    save_recording(frames)

# Modify array of frames
def start_recording(img, frames):
    
    # Convert image into np array
    frame = np.asarray(img)
    frames.append(frame)
    return frames

# Save frames to mp4 file
def save_recording(frames):
    
    imageio.mimwrite(os.path.join(save_path,'test.mp4'), frames , fps = 2)
    print("Recording saved as '{}'".format('test.mp4'))


main()