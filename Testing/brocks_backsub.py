# backsub_w_contour.py
from __future__ import print_function
import time
import cv2 as cv
import argparse
import os
import numpy as np
import imageio
import scipy.ndimage as sp
import pdb

main_path = os.path.dirname(os.path.abspath(__file__)) 
grab_path = os.path.join(main_path, "preprocess")
addr = os.path.join(grab_path, "test2.mp4")
save_path = os.path.join(main_path, "postprocess2")

def main():
    video = cv.VideoCapture(addr)

    kernel = None

    background_object = cv.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=True)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        fgmask = background_object.apply(frame)
        
        _, fgmask = cv.threshold(fgmask, 250, 255, cv.THRESH_BINARY)

        fgmask = cv.erode(fgmask, kernel=kernel, iterations=3)
        fgmask = cv.dilate(fgmask, kernel=kernel, iterations=10)
        
        # cv.imshow("", fgmask)
        # cv.waitKey(1)
        
        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        frame_copy = frame.copy()
        for c in contours:
            # 
            if cv.contourArea(c) > 16000:
                x, y, width, height = cv.boundingRect(c)
                cv.rectangle(frame_copy, (x,y), (x+width, y+height), (0,0,255), 2)
                cv.putText(frame_copy, "car detected", (x,y-10), cv.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)
        foregound_part = cv.bitwise_and(frame, frame, mask=fgmask)
        stacked = np.hstack((frame, foregound_part, frame_copy))
        cv.imshow("", cv.resize(stacked, None, fx=0.5, fy=0.5))
        cv.waitKey(100)

        start_recording(foregound_part, frames)
    video.release()
    cv.destroyAllWindows()
    #save_recording(frames)

# Modify array of frames
def start_recording(img, frames):
    
    # Convert image into np array
    frame = np.asarray(img)
    frames.append(frame)
    return frames

# Save frames to mp4 file
def save_recording(frames):
    
    imageio.mimwrite(os.path.join(save_path,'test4.mp4'), frames , fps = 2)
    print("Recording saved as '{}'".format('test4.mp4'))

main()