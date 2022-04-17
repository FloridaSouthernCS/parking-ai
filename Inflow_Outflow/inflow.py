# inflow.py

from turtle import left
import cv2 
import os
import pdb
import numpy as np 
import math
import imutils
from shapely.geometry import Point, Polygon
from optic_flow import lk_optic_flow
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
    It repeats this process to group contour areas together'''

    # Find contours twice to merge all contours that touch
    cmask, contours = find_and_draw_contours(frame, fgmask)
    cmask = contours_to_foreground_mask(np.array(cmask), (0,255,0))     
    cmask, contours = find_and_draw_contours(frame, cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))
    cmask = contours_to_foreground_mask(cmask, (255,255,255))
    _, contours = find_and_draw_contours(frame, cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))
    # cv2.imshow('a', _)
    # cv2.waitKey()
    # print("contour num: ", len(contours))

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

def get_tracking_points(frame, contours, count):
    '''This method grabs the left, right, top, and bottom points of a contour area'''
    
    points_frame = frame.copy()

    # If we have contours, choose the contours we want and draw them
    extreme_points = []
    if (len(contours) > 0):
        
        extreme_points = get_extreme_points(contours) # grab extreme points on contours 

        for point in extreme_points: # draw the extreme points being tracked
            points_frame = cv2.circle(points_frame, point, 8, (255, 0, 0), -1)

    points_frame = draw_polygon(points_frame) # draw area of interest 
    extreme_points, count = get_valid_points(extreme_points, count)

    return points_frame, extreme_points, count

def draw_points(frame, points):
    draw_frame = frame.copy()
    
    for point in points:
        draw_frame = cv2.circle(draw_frame, point, 8, (255, 0, 0), -1)
    return draw_frame


def get_extreme_points(contours):
    '''Grab the extreme right, left, top, and bottom points of the contour area'''
    points = max(contours, key=cv2.contourArea) # grab all max points
    
    # get top, bottom, left, right points
    left_point = tuple(points[points[:, :, 0].argmin()][0])
    right_point = tuple(points[points[:, :, 0].argmax()][0])
    top_point = tuple(points[points[:, :, 1].argmin()][0])
    bottom_point = tuple(points[points[:, :, 1].argmax()][0])

    return np.array([left_point, right_point, top_point, bottom_point]) 

def get_valid_points(points, count):
    '''This counts the number of points that have hit the left side'''
    for i in range(len(points)):
        if points[i][0] <= 10:
            count += 1

    return points, count

def keep_tracking_points(tracking_points, point_count, tracking_points_threshold, keep_tracking):
    '''This method controls if the points should be tracked using optic flow'''
    # if there's at least 3 points that has hit the left of the frame stop tracking
    if point_count >= tracking_points_threshold: 
        return False
    # if motion is detected in area of interest start tracking 
    elif motion_detected_in_area_of_interest(tracking_points): 
        return True 
    return keep_tracking



def draw_polygon(points_frame):
    '''Draws the Polygon Area of Interest on the frame'''
    
    # display start area of interest in red 
    points = np.array([[600, 150], [950, 380], [1023, 300], [1023, 170],[850, 100]], np.int32).reshape((-1, 1, 2))
    cv2.polylines(points_frame, [points], True, (0,0,255), 2)

    return points_frame 

def motion_detected_in_area_of_interest(points):
    '''This method tests if the right most point of a contour area entered into the area of iterest'''
    for point in points:
        point = Point(point)
        area_of_interest = Polygon([(600, 50), (500, 450), (1000, 450), (1000, 50)])
        if area_of_interest.contains(point): return True
    return False 

def get_optic_flow(lk_flow, cmask, frame, tracking_points):

    '''Calls the optic flow function to track the tracking_points'''
    flow_img = frame.copy()
    if not type_null(tracking_points): 
        lk_flow.set_mask(cmask)
        flow_img = lk_flow.get_flow(flow_img, tracking_points)

    return flow_img

def type_null(array):
    is_null = type(array) == type(None) or array == []
    return is_null

def format_window(frames, max_h_frames, max_width):
    '''This method is used to format the final output window
    PARAMETERS: 
    - frames: a list of all the frames which are to be displayed
    - max_h_frames: maximum horizontal frames to be displayed
    - mad_width: maximum desired pixel width for the output window '''

    filler = zeros_frame(frames[0])
    
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
