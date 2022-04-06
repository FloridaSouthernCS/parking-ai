# backsub_w_contour.py
# from lib2to3.pgen2.token import frameAL
# from logging import captureWarnings
# from operator import xor
from turtle import left
import cv2 
import os
# from cv2 import threshold
# import scipy.ndimage as sp
import pdb
import numpy as np 
import tkinter as tk
import math
import key_log
import record
from optic_flow import lk_optic_flow
import imutils

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# FILEPATHS 
main_path = os.path.dirname(os.path.abspath(__file__)) 

save_folder = "Inflow_Results"
save_path = os.path.join(main_path, save_folder)

datapath = os.path.join(main_path, "Data", "Inflow")

car_path = os.path.join(datapath, "Car")
combo_path = os.path.join(datapath, "Combo")
not_car_path = os.path.join(datapath, "Not_Car")

# addr = os.path.join(car_path, "car1.mp4")
# addr = os.path.join(combo_path, "combo5.mp4")
# addr = os.path.join(not_car_path, "not_car10.mp4")



addr = os.path.join(car_path, "car6.mp4")
addr = os.path.join(car_path, "car10.mp4")
# addr = os.path.join(combo_path, "combo7.mp4")
# addr = os.path.join(not_car_path, "not_car11.mp4")


# PARAMETERS
VAR_THRESHOLD = 200
CONTOUR_THRESHOLD = 7000 # COUNTOR THRESHOLD FOR CONTOUR AREA


# KANADE PARAMETERS
# params for corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
  
# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Get screen resolution
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()


def get_cmask(contour_frame, frame):
    cmask = contour_mask(np.array(contour_frame), (0,255,0))
            
    _, cmask, frame_norm = c_hull(frame, cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))
    cmask = contour_mask(cmask, (255,255,255))

    foreground = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))

    # return foreground, cmask
    return foreground, cmask, frame_norm
    


'''This method applies Background Subtraction 
PARAMETERS: 
- frame: frame from video
- background_object: filter to apply to frame from background subtraction''' 
def back_sub(frame, background_object):
    fgmask = frame.copy()
    temp = frame.copy()

    fgmask = background_object.apply(fgmask) # apply background subtraction to frame 
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # remove the gray pixels which represent shadows

    fgmask = cv2.dilate(fgmask, kernel=(25,25), iterations=5) # dilate
    foreground = cv2.bitwise_and(temp, temp, mask=fgmask) # show frame in areas in motion

    return fgmask, foreground
    
 
'''This method draws the rectangles around areas of detected motion
PARAMETERS: 
- frame: frame from the video
- fgmask: foreground mask to show which areas motion was detected '''
def contour_detection(frame, fgmask):

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
    contour_frame = frame.copy()

    for i in range(len(contours)): # for each contour found 
        if cv2.contourArea(contours[i]) > CONTOUR_THRESHOLD: # if the countour area is above a # 

            cv2.drawContours(contour_frame, contours, i, (0,255,0), 3)

            # # draw a rectangle 
            # x, y, width, height = cv2.boundingRect(contours[i])
            # cv2.rectangle(contour_frame, (x,y - 10), (x + width, y + height), (0,0,255), 2)
            # cv2.putText(contour_frame, "car detected", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    return None, contour_frame 

def contour_approx(frame, fgmask):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
    contour_frame = frame.copy()

    for c in contours:
        if cv2.contourArea(c) > CONTOUR_THRESHOLD:
            epsilon = 0.01*cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(contour_frame, [approx], 0, (0, 255, 0), 3)
    return None, contour_frame

def contour_hull(frame, fgmask):

    contour_frame = frame.copy()
    extLeft = 0
    extRight = 0
    flag = False 

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contourss
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2. CHAIN_APPROX_SIMPLE)

    cnts= imutils.grab_contours(cnts)
    
    

    # pdb.set_trace()
    count = 0
    for c in contours:
        if cv2.contourArea(c) > CONTOUR_THRESHOLD:
            flag = True 
            hull = cv2.convexHull(c)
            count += 1
            # saves.append(hull)
            cv2.drawContours(contour_frame, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)

    if (len(cnts) > 0 and flag):
        ce= max(cnts, key=cv2.contourArea)

        extLeft = tuple(ce[ce[:, :, 0].argmin()][0])
        extRight = tuple(ce[ce[:, :, 0].argmax()][0])
        extTop = tuple(ce[ce[:, :, 1].argmin()][0])
        extBot = tuple(ce[ce[:, :, 1].argmax()][0])
        cv2.circle(contour_frame, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(contour_frame, extRight, 8, (0, 255, 0), -1)
        cv2.circle(contour_frame, extTop, 8, (255, 0, 0), -1)
        cv2.circle(contour_frame, extBot, 8, (255, 255, 0), -1)
        
    # return None, contour_frame
    return None, contour_frame,  extRight, extLeft, frame

def c_hull(frame, fgmask):

    contour_frame = frame.copy()
    extLeft = 0

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contourss
    # cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2. CHAIN_APPROX_SIMPLE)

    # cnts= imutils.grab_contours(cnts)
    # if (len(cnts) > 0):
    #     ce= max(cnts, key=cv2. contourArea)

    #     extLeft = tuple(ce[ce[:, :, 0].argmin()][0])
    #     extRight = tuple(ce[ce[:, :, 0].argmax()][0])
    #     extTop = tuple(ce[ce[:, :, 1].argmin()][0])
    #     extBot = tuple(ce[ce[:, :, 1].argmax()][0])
    #     cv2.circle(contour_frame, extLeft, 8, (0, 0, 255), -1)
    #     cv2.circle(contour_frame, extRight, 8, (0, 255, 0), -1)
    #     cv2.circle(contour_frame, extTop, 8, (255, 0, 0), -1)
    #     cv2.circle(contour_frame, extBot, 8, (255, 255, 0), -1)
    

    # pdb.set_trace()
    count = 0
    for c in contours:
        if cv2.contourArea(c) > CONTOUR_THRESHOLD:
            hull = cv2.convexHull(c)
            count += 1
            # saves.append(hull)
            cv2.drawContours(contour_frame, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
    
    # return None, contour_frame
    return None, contour_frame, frame

def contour_mask(contour, color):
    
    r1, g1, b1 = color # Original value
    r2, g2, b2 = 0, 0, 0 # Value that we want to replace it with

    red, green, blue = contour[:,:,0], contour[:,:,1], contour[:,:,2]
    bleh = ~ ((red == r1) | (green == g1) | (blue == b1))
    contour[:,:,:3][bleh] = [r2, g2, b2]

    temp = cv2.cvtColor(contour, cv2.COLOR_RGB2GRAY)
    _, fgmask = cv2.threshold(temp, 1, 255, cv2.THRESH_BINARY)
    
    
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    return fgmask


'''This method is used to format the final output window
PARAMETERS: 
- frames: a list of all the frames which are to be displayed
- max_h_frames: maximum horizontal frames to be displayed
- mad_width: maximum desired pixel width for the output window '''
def format_window(frames, max_h_frames, max_width):

    filler = np.zeros(np.asarray(frames[0]).shape,dtype=np.uint8 )
    
    single_frame_ratio = frames[0].shape[0]/frames[0].shape[1]

    frame_count = len(frames)
    
    filler_count = 0
    if frame_count%max_h_frames != 0:
        filler_count = max_h_frames - (frame_count%max_h_frames)
        
        frames = np.hstack(frames)
        for i in range(filler_count):
            frames = np.hstack([frames, filler])
    else:
        frames = np.hstack(frames)

    #pdb.set_trace()
    frames = np.hsplit(frames, filler_count+frame_count)
    hframeslist = []
    
    for i in range(0, len(frames), max_h_frames):
        hframeslist.append(np.hstack(frames[i:i+max_h_frames]))
        
    window = np.vstack(hframeslist[:])

    ratio = window.shape[0]/window.shape[1]
    
    window = cv2.resize(window, dsize=(math.floor( max_width ), math.floor( max_width*ratio )) )
    
    return window
        

'''This method acts as a controller to manipulate the video feed
PARAMETERS: 
- logger: log object from key_log.py '''
def check_log(logger, recording):
    
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