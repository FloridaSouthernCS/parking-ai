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
grab_path = os.path.join(main_path, "Preprocess\\Inflow\\Car")
addr = os.path.join(grab_path, "car11.mp4")
save_path = os.path.join(main_path, "postprocess2")

def main():
    video = cv.VideoCapture(addr)

    kernel = None

    background_object = cv.createBackgroundSubtractorMOG2(varThreshold=200, detectShadows=True)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        

        fgmask = get_fgmask(background_object, frame)
        
        contour_frame = get_contours(frame, fgmask)

        foregound_part = cv.bitwise_and(frame, frame, mask=fgmask)

        stacked = np.hstack((frame, foregound_part, contour_frame))

        cv.imshow("", cv.resize(stacked, None, fx=0.5, fy=0.5))
        cv.waitKey(50)
        
    video.release()
    
    


def get_contours(frame, fgmask):
    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_frame = frame.copy()
    for c in contours:
        if cv.contourArea(c) > 40000:
            x, y, width, height = cv.boundingRect(c)
            cv.rectangle(contour_frame, (x,y), (x+width, y+height), (0,0,255), 2)
            cv.putText(contour_frame, "car detected", (x,y-10), cv.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)
    return contour_frame

def get_fgmask(subtractor, frame ):
    fgmask = sp.gaussian_filter(frame, sigma = 4)
    fgmask = subtractor.apply(frame)
    fgmask = cv.erode(fgmask, kernel=(10,10), iterations=2)
    fgmask = sp.gaussian_filter(fgmask, sigma = 4)
    _, fgmask = cv.threshold(fgmask, 50, 255, cv.THRESH_BINARY)

    fgmask = cv.dilate(fgmask, kernel=None, iterations=20)
    # cv.imshow("", fgmask)
    # cv.waitKey(1)
    fgmask = sp.gaussian_filter(fgmask, sigma = 2.7)

    return fgmask

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

if __name__ == "__main__":
    main()