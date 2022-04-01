# inflow_main.py

import cv2 
import os

import numpy as np 
import tkinter as tk
import key_log
import record
from optic_flow import lk_optic_flow
from inflow import *


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
    


    try:
        
        ret, frame = cap.read()
        old_frame = cv2.normalize(frame, frame, 0, 220, cv2.NORM_MINMAX)
        lk_flow = lk_optic_flow(old_frame, feature_params, lk_params)
        
        p0 = []
        cmask = []
        while True:
            
            display_frames = []

            '''Extract image from input mp4 video file'''
            ret, frame = cap.read()
            if not ret: break
            display_frames.append(frame) 


            '''Background subtraction to detect motion'''
            # # Get binary mask of movement
            backsub_mask1, backsub_frame1 = back_sub(frame, background_object)


            '''Contour Detection with threshold to find reigons of interest'''
            # Get an enhanced mask by thresholding reigons of interest by sizes of white pixel areas
            # contour_detection_crop, contour_detection_frame = contour_detection(frame, backsub_mask)
            # display_frames.append(contour_detection_frame) 

            # contour_approx_crop, contour_approx_frame = contour_approx(frame, backsub_mask)
            # display_frames.append(contour_approx_frame) 

            # contour_hull_crop, contour_hull_frame = contour_hull(frame, backsub_mask)
            # display_frames.append(contour_hull_frame) 


            frame_norm = cv2.normalize(frame, frame, 0, 220, cv2.NORM_MINMAX)

            # grabbing area of interest (contour area should start tracking)
            bounding_rect = frame_norm.copy()
            cv2.rectangle(bounding_rect, (500,50), (1000,450), (0,0,255), 2)

            # background subtraction
            backsub_mask, backsub_frame = back_sub(frame_norm, background_object)

            # contour areas 
            contour_crop, contour_frame, right_point, left_point = contour_hull(frame_norm, backsub_mask)

            # Get double convex hulled max
            # if len(cmask) <= 0:
            foreground, cmask = get_cmask(contour_frame, frame)
            flow_img = np.empty(cmask.shape)

            

            # point_of_interest = frame[500:1000, 50:250] #where we want to detect the initial contour area
            
            # check if left most point of rectangle sis in the area of interest 
            # if right_point != 0:
            #     point = Point(right_point)
            #     area_of_interest = Polygon([(500, 50), (500, 450), (1000, 450), (1000, 50)])
            #     print(area_of_interest.contains(point)) # condition for starting optic flow
            if left_point != 0: 
                lk_flow.set_mask(cmask)
                flow_img = lk_flow.get_flow(frame_norm.copy(), right_point, left_point)

                cv2.rectangle(contour_frame, (500,50), (1000,450), (0,0,255), 2)
                display_frames = np.asarray([frame_norm, cv2.cvtColor(backsub_mask, cv2.COLOR_GRAY2BGR), contour_frame, foreground, flow_img])#frame,  cv2.cvtColor(backsub_mask2, cv2.COLOR_GRAY2BGR), contour_frame4])
            else: 
                cv2.rectangle(contour_frame, (500,50), (1000,450), (0,0,255), 2)

                display_frames = np.asarray([frame_norm, cv2.cvtColor(backsub_mask, cv2.COLOR_GRAY2BGR), contour_frame, foreground])#frame,  cv2.cvtColor(backsub_mask2, cv2.COLOR_GRAY2BGR), contour_frame4])

            '''Display output in a practical way'''
            # USE THIS VARIABLE TO WRAP THE WINDOW
            max_h_frames = 3
            # Format the output
            window = format_window(display_frames, max_h_frames, screen_width*.75)
            
            if len(frame) > 0:
                old_frame = frame

            # Show image
            cv2.imshow("", window)
            cv2.waitKey(50)
            
            # Check if we should still be recording (and other controls)
            recording = check_log(logger, recording)

            if recording == True:
                record.start_recording(window, frames)
            
        logger.stop()
    except Exception as e:
        logger.stop()
        raise e
    

    if frames != []:
        record.save_recording(frames, save_path, "inflow_results")
    
    cv2.destroyAllWindows() 
    cap.release()
    

if __name__ == '__main__':
    main()