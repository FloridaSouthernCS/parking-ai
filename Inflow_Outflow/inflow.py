# inflow.py
# Holds many of the functions used for inflow_main

import cv2 
import numpy as np 
import math
import imutils
from shapely.geometry import Point, Polygon
from inflow import *

# PARAMETERS
CONTOUR_THRESHOLD = 7000 # COUNTOR THRESHOLD FOR CONTOUR AREA

def back_sub(fgmask, background_object):
    '''This method applies Background Subtraction 
    PARAMETERS: 
    - frame: frame from video
    - background_object: filter to apply to frame from background subtraction''' 

    temp = fgmask.copy()

    fgmask = background_object.apply(fgmask) # apply background subtraction to frame 
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # remove the gray pixels which represent shadows

    fgmask = cv2.dilate(fgmask, kernel=(25,25), iterations=5) # dilate
    foreground = cv2.bitwise_and(temp, temp, mask=fgmask) # show frame in areas in motion

    return fgmask, foreground


def get_cmask(fgmask, frame):
    '''This method finds the contour areas and grab a convex hull around the contour areas. 
    It repeats this process to group contour areas together
    
    IF ANYONE IS READING THIS IN THE FUTURE, THIS IS A POOR IMPLEMENTATION. THIS FUNCTION FINDS CONTOURS, DRAWS THEM,
    THEN USES THE DRAWN OUTPUT AS INPUT. IN REALITY, NO DRAWING SHOULD OCCUR UNTIL THE END.
    '''

    # Find contours thrice to merge all contours that touch
    cmask, contours = find_and_draw_contours(frame, fgmask)
    cmask = contours_to_foreground_mask(np.array(cmask), (0,255,0))     
    cmask, contours = find_and_draw_contours(frame, cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))
    cmask = contours_to_foreground_mask(cmask, (255,255,255))
    _, contours = find_and_draw_contours(frame, cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))

    # Show the mask applied to the original frame. Car in grayscale should appear amongst a sea of black pixels.
    foreground = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))

    return foreground, cmask, contours

def find_and_draw_contours(frame, fgmask):
    '''This method finds the contour areas a draws them is they're above a contour area threshold'''
    contour_frame = frame.copy()

    contours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contourss
    cnts = imutils.grab_contours(contours)
    
    for c in cnts:
        if cv2.contourArea(c) > CONTOUR_THRESHOLD:
            hull = cv2.convexHull(c)            
            cv2.drawContours(contour_frame, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
    
    return contour_frame, cnts

def contours_to_foreground_mask(contour, color):
    '''This method turns the contours areas into a foreground mask so they can be grouped'''
    r1, g1, b1 = color # Original value
    r2, g2, b2 = 0, 0, 0 # Value that we want to replace it with

    red, green, blue = contour[:,:,0], contour[:,:,1], contour[:,:,2]
    bleh = ~ ((red == r1) | (green == g1) | (blue == b1))
    contour[:,:,:3][bleh] = [r2, g2, b2]

    temp = cv2.cvtColor(contour, cv2.COLOR_RGB2GRAY)
    _, fgmask = cv2.threshold(temp, 1, 255, cv2.THRESH_BINARY)
    
    
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    return fgmask

def zeros_frame(frame):
    filler = np.zeros(np.asarray(frame).shape, dtype=np.uint8 )
    return filler

def format_window(frames, max_h_frames, max_width):
    '''This method is used to format the final output window
    PARAMETERS: 
    - frames: a list of all the frames which are to be displayed
    - max_h_frames: maximum horizontal frames to be displayed
    - mad_width: maximum desired pixel width for the output window '''

    filler = zeros_frame(frames[0])
    
    frame_count = len(frames)
    
    filler_count = 0
    if frame_count%max_h_frames != 0:
        filler_count = max_h_frames - (frame_count%max_h_frames)
        
        frames = np.hstack(frames)
        for i in range(filler_count):
            frames = np.hstack([frames, filler])
    else:
        frames = np.hstack(frames)


    frames = np.hsplit(frames, filler_count+frame_count)
    hframeslist = []
    
    for i in range(0, len(frames), max_h_frames):
        hframeslist.append(np.hstack(frames[i:i+max_h_frames]))
        
    window = np.vstack(hframeslist[:])

    ratio = window.shape[0]/window.shape[1]
    
    window = cv2.resize(window, dsize=(math.floor( max_width ), math.floor( max_width*ratio )) )
    
    return window
        

def check_log(logger, recording):
    '''This method acts as a controller to manipulate the video feed
    PARAMETERS: 
    - logger: log object from key_log.py '''
    
    if logger.keys_clicked:
        key = logger.keys_clicked[-1]
        
        if key in logger.valid_keys:
            
            if key == 's':
                logger.keys_clicked.append(None)
                key = logger.keys_clicked[-1]
                while 's' != key:
                    key = logger.keys_clicked[-1]
                    
                    cv2.waitKey(50)
                logger.keys_clicked.append(None)
            elif key == 'r':
                recording = not recording
                cv2.waitKey(50)
                logger.keys_clicked.append(None)
            elif key == 'q':
                recording = not recording
                cv2.waitKey(50)
                logger.keys_clicked.append(None)
                raise

            else:
                print("The key {" + key + "} has not been set up. Set up this key in 'check_log'")
    return recording
