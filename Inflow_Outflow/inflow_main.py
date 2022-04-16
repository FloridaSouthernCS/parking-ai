# inflow_main.py

from sys import displayhook
import cv2 
import os

import numpy as np 
import tkinter as tk
import key_log
import record
from optic_flow import lk_optic_flow
from inflow import *
from Trackable import Trackable
from Trackable_Manager import Trackable_Manager


'''
    Document conventions:
    - A variable that contains this in its name: "frame" suggests that it contains the best visualization for output.
    This means it will always include the original input image and apply some visualization to it.
    
'''


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

# addr = os.path.join(car_path, "car1.mp4")
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


def main():
    # Create key_log object to control what video is processed
    print("Controls: ")
    print("  s: Start/Stop")
    print("  r: Record Start/Pause")
    # print("  a: Previous")
    # print("  d: Next")
    logger = key_log.log(['s', 'r', 'q'])
    logger.start()

    # Get Video from mp4 source
    cap = cv2.VideoCapture(addr)

    frames = []
    recording = False

    # Get R.O.I. tool
    background_object = cv2.createBackgroundSubtractorMOG2(varThreshold=VAR_THRESHOLD, detectShadows=False) 
    # prev_points = np.full((4,2), None)
    point_count = 0
    keep_tracking = False 


    try:
        
        ret, frame = cap.read()
        old_frame = cv2.normalize(frame, frame, 0, 220, cv2.NORM_MINMAX)
        lk_flow = lk_optic_flow(old_frame, feature_params, lk_params)

        track_man = Trackable_Manager(frame)
        
        p0 = []
        cmask = []
        while True:
            
            ''' 
            FRAME 1
            Raw input 
            '''
            ret, frame = cap.read()
            if not ret: break

            ''' 
            FRAME 2 and 3
            Background subtraction and Frame normalization
            '''
            backsub_mask1, backsub_frame1 = back_sub(frame.copy(), background_object)
            frame_norm = cv2.normalize(frame, frame, 0, 220, cv2.NORM_MINMAX)
            backsub_mask_grey, backsub_frame = back_sub(frame_norm, background_object)
            # pdb.set_trace()
            ''' 
            FRAME 4
            Get Contours 
            '''
            contour_foreground, cmask, contours = get_cmask(backsub_mask_grey, frame)
            
            ''' 
            FRAME 5
            Get Extreme points 
            '''
            points_frame, tracking_points, point_count = get_tracking_points(frame_norm, contours, point_count) # Isolate extreme points from contours
            
            '''
            FRAME 6 
            Track Movement over time of extreme points
            '''
            # determine if we keep tracking the points 
            keep_tracking = keep_tracking_points(tracking_points, point_count, 3, keep_tracking)

            if keep_tracking:
                flow_img = get_optic_flow(lk_flow, cmask, frame_norm.copy(), tracking_points)
            else:
                flow_img = frame_norm.copy()

            '''
            FRAME 7
            Get Center points of contours
            '''
            center = centers_of_contours(contours)
            center_frame = draw_points(frame_norm.copy(), center)
            

            '''
            Display the frames
            '''
            track_man.set_frame(frame_norm)
            trackables = track_man.get_trackables_from_contours(contours)
            track_man.add_trackables(trackables)
            
            track_frame = track_man.get_trackable_frame()

            display_frames = np.asarray([frame, frame_norm, backsub_frame, contour_foreground, points_frame, flow_img, center_frame, track_frame]) # display frames 

            # Format window output
            max_h_frames = 3
            window = format_window(display_frames, max_h_frames, screen_width*.75)

            # Show image
            cv2.imshow("", window)
            cv2.waitKey(50)


            ''' Update loop variables '''
            if len(frame) > 0:
                old_frame = frame_norm



            ''' Record if requested '''
            # Check if we should still be recording (and other controls)
            recording = check_log(logger, recording)

            if recording == True:
                record.start_recording(window, frames)


            
        logger.stop()
    except Exception as e:
        logger.stop()
        raise e
    
    ''' Save Recording if present '''
    if frames != []:
        record.save_recording(frames, save_path, "inflow_results")
    
    ''' Stop Capture '''
    cv2.destroyAllWindows() 
    cap.release()
    

if __name__ == '__main__':
    main()