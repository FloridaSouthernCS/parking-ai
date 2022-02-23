# backsub_w_contour.py
from operator import xor
import cv2 
import os
import scipy.ndimage as sp
import pdb
import numpy as np 
import tkinter as tk
import math
import key_log

'''
    Document conventions:
    - A variable that contains this in its name: "frame" suggests that it contains the best visualization for output.
    This means it will always include the original input image and apply some visualization to it.
    
'''


# FILEPATHS 
main_path = os.path.dirname(os.path.abspath(__file__)) 
datapath = os.path.join(main_path, "Data", "Inflow")

car_path = os.path.join(datapath, "Car")
combo_path = os.path.join(datapath, "Combo")
not_car_path = os.path.join(datapath, "Not_Car")

addr = os.path.join(combo_path, "combo5.mp4")


# PARAMETERS
VAR_THRESHOLD = 50 # BACKGROUND SUB PARAMETER

CONTOUR_THRESHOLD = 5000 # COUNTOR THRESHOLD FOR CONTOUR AREA


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

def main():
    # Create key_log object to control what video is processed
    print("Controls: ")
    print("  s: Start/Stop")
    # print("  a: Previous")
    # print("  d: Next")
    logger = key_log.log(['s'])
    logger.start()

    # Get Video from mp4 source
    cap = cv2.VideoCapture(addr)
    
    # Get R.O.I. tool
    background_object = cv2.createBackgroundSubtractorMOG2(varThreshold=VAR_THRESHOLD, detectShadows=True) 
    
    while True:

        display_frames = []

        '''Extract image from input mp4 video file'''
        ret, frame = cap.read()
        if not ret: break
        display_frames.append(frame) 

        '''Background subtraction to detect motion'''
        # Get binary mask of movement
        backsub_mask, backsub_frame = back_sub(frame, background_object)
        display_frames.append(backsub_frame) 


        '''Contour Detection with threshold to find reigons of interest'''
        # Get an enhanced mask by thresholding reigons of interest by sizes of white pixel areas
        contour_crop, contour_frame = contour_detection(frame, backsub_mask)
        display_frames.append(contour_frame) 


        '''Feature Detection with Kernel Convolution'''
        # Get array of points where Kernel Convolution was most effective
        features, features_frame = feature_detection(frame, contour_crop)
        display_frames.append(features_frame) 


        '''Feature Segmentation using Clustering'''
        # Segment the features into clusters to best imply the existence of individual vehicles
        clusters, clustering_frame = clustering(frame, features)
        display_frames.append(clustering_frame) 


        '''Feature Motion using Optic Flow'''
        # Track the motion of each feature
        feature_motions, feature_motions_frame = track_features(frame, clusters)
        display_frames.append(feature_motions_frame) 
        # Track the motion of each cluster (cluster motion found using the average of each features' motion in a given cluster)
        cluster_motions, cluster_motions_frame = track_clusters(frame, feature_motions)
        display_frames.append(cluster_motions_frame) 



        ''' IMPLEMENTATION THOUGHTS AND IDEAS:
            - If a feature has motion that deviates from it's cluster too much, it should be discarded as a feature worth tracking
            
        '''


        '''Display output in a practical way'''

        display_frames = np.array([frame, backsub_frame, contour_frame])

        max_h_frames = 3
        window = format_window(display_frames, max_h_frames, screen_width)
        
        
        cv2.imshow("", window)
        cv2.waitKey(50)
        
        check_log(logger)

    logger.stop()
    cv2.destroyAllWindows()
    cap.release()


'''This method applies Background Subtraction 
PARAMETERS: 
- frame: frame from video
- background_object: filter to apply to frame from background subtraction''' 
def back_sub(frame, background_object):

    fgmask = background_object.apply(frame) # apple background subtraction to frame 
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # grab just the blacker parts of teh fgmask 
    foregound = cv2.bitwise_and(frame, frame, mask=fgmask) # show frame in areas in motion

    return fgmask, foregound


    # OTHER METHODS
    # sp.gaussian_filter(frame, sigma = 4) # blur
    # cv2.erode(fgmask, kernel=(10,10), iterations=2) # erode
    # _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # apply threshold
    # fgmask = cv2.dilate(fgmask, kernel=None, iterations=30) # dilate

 
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

def feature_detection(initial_frame, frame):
    return None, None

def clustering(initial_frame, frame):
    return None, None

def track_features(initial_frame, frame):
    return None, None

def track_clusters(initial_frame, frame):
    return None, None


'''This method is used to format the final output window
PARAMETERS: 
- frames: a list of all the frames which are to be displayed
- max_h_frames: maximum horizontal frames to be displayed
- mad_width: maximum desired pixel width for the output window '''
def format_window(frames, max_h_frames, max_width):

    window = np.hstack(frames[:])

    ratio = window.shape[0]/window.shape[1]
    
    window = cv2.resize(window, dsize=(math.floor( max_width ), math.floor( max_width*ratio )) )
    
    return window
        

'''This method acts as a controller to manipulate the video feed
PARAMETERS: 
- logger: log object from key_log.py '''
def check_log(logger):
    
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
            else:
                print("The key {" + key + "} has not been set up. Set up this key in 'check_log'")




if __name__ == "__main__":
    main()