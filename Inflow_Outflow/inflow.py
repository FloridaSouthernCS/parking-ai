# backsub_w_contour.py
import cv2 
import os
import numpy as np
import scipy.ndimage as sp
import pdb

main_path = os.path.dirname(os.path.abspath(__file__)) 
datapath = os.path.join(main_path, "Data", "Inflow")

car_path = os.path.join(datapath, "Car")
combo_path = os.path.join(datapath, "Combo")
not_car_path = os.path.join(datapath, "Not_Car")

addr = os.path.join(car_path, "car11.mp4")

VAR_THRESHOLD = 200 # BACKGROUND SUB PARAMETER


'''This method applies Background Subtraction 
PARAMETERS: 
- frame: frame from video
- background_object: filter to apply to frame from background subtraction''' 
def back_sub(frame, background_object):

    fgmask = background_object.apply(frame) # apple background subtraction to frame 
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # grab just the blacker parts of teh fgmask 
    foregound_part = cv2.bitwise_and(frame, frame, mask=fgmask) # show frame in areas in motion

    return foregound_part


    # OTHER METHODS
    # sp.gaussian_filter(frame, sigma = 4) # blur
    # cv2.erode(fgmask, kernel=(10,10), iterations=2) # erode
    # _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # apply threshold
    # fgmask = cv2.dilate(fgmask, kernel=None, iterations=30) # dilate 

    
    
def main():
    vid = cv2.VideoCapture(addr)

    background_object = cv2.createBackgroundSubtractorMOG2(varThreshold=VAR_THRESHOLD, detectShadows=True) 

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        foregound_part = back_sub(frame, background_object)

        cv2.imshow("", foregound_part)
        cv2.waitKey(50)
        
    vid.release()

    


if __name__ == "__main__":
    main()