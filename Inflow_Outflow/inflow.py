# backsub_w_contour.py
# from lib2to3.pgen2.token import frameAL
# from logging import captureWarnings
# from operator import xor
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



addr = os.path.join(car_path, "car10.mp4")
# addr = os.path.join(combo_path, "combo7.mp4")
# addr = os.path.join(not_car_path, "not_car11.mp4")


# PARAMETERS
VAR_THRESHOLD = 200 # BACKGROUND SUB PARAMETER

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
        p0 = []
        cmask = []
        while True:
            
            display_frames = []

            '''Extract image from input mp4 video file'''
            ret, frame = cap.read()
            if not ret: break
            display_frames.append(frame) 
            frame = cv2.normalize(frame, frame, 0, 220, cv2.NORM_MINMAX)
            '''Background subtraction to detect motion'''
            # Get binary mask of movement
            backsub_mask, backsub_frame = back_sub(frame, background_object)
            display_frames.append(backsub_frame) 


            '''Contour Detection with threshold to find reigons of interest'''
            # Get an enhanced mask by thresholding reigons of interest by sizes of white pixel areas
            # contour_det_crop, contour_det = contour_detection(frame, backsub_mask)
            # display_frames.append(contour_det) 

            # contour_app_crop, contour_app = contour_approx(frame, backsub_mask)
            # display_frames.append(contour_app) 

            # contour_hull_crop, contour_hul = contour_hull(frame, backsub_mask)
            # display_frames.append(contour_hul)  

            # contour areas 
            contour_crop4, contour_frame4 = contour_hull(frame, backsub_mask)
            display_frames.append(contour_frame4) 

            
            
            # Get double convex hulled max
            if len(cmask) <= 0:
                foreground, cmask = get_cmask(contour_frame4, frame)
                flow_img = np.empty(cmask.shape)
            # Apply optic flow to foreground of cmask
            flow_img, p0 = optic_flow(frame, old_frame, cmask, p0)
                


            display_frames = np.asarray([frame, cv2.cvtColor(backsub_mask, cv2.COLOR_GRAY2BGR), contour_frame4, foreground, flow_img ])#frame,  cv2.cvtColor(backsub_mask2, cv2.COLOR_GRAY2BGR), contour_frame4])

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
        raise
    

    if frames != []:
        record.save_recording(frames, save_path, "inflow_results")
    
    cv2.destroyAllWindows() 
    cap.release()


def get_cmask(contour_frame, frame):
    cmask = contour_mask(np.array(contour_frame), (0,255,0))
            
    _, cmask = contour_hull(frame, cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))
    # pdb.set_trace()
    cmask = contour_mask(cmask, (255,255,255))

    foreground = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(cmask, cv2.COLOR_RGB2GRAY))

    return foreground, cmask
    


'''This method applies Background Subtraction 
PARAMETERS: 
- frame: frame from video
- background_object: filter to apply to frame from background subtraction''' 
def back_sub(frame, background_object):
    fgmask = frame

    # fgmask = apply_pyramids(fgmask, 1)
    
    fgmask = background_object.apply(fgmask) # apply background subtraction to frame 
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # remove the gray pixels which represent shadows
    # fgmask = cv2.erode(fgmask, kernel=(15,15), iterations = 5)

    fgmask = cv2.dilate(fgmask, kernel=(25,25), iterations=5) # dilate
    foregound = cv2.bitwise_and(frame, frame, mask=fgmask) # show frame in areas in motion

    return fgmask, foregound


    # OTHER METHODS
    # sp.gaussian_filter(frame, sigma = 4) # blur
    # cv2.erode(fgmask, kernel=(10,10), iterations=2) # erode
    # _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) # apply threshold
    # fgmask = cv2.dilate(fgmask, kernel=None, iterations=30) # dilate


'''This method reduces noise by applying gaussian pyramids. Pyramid up and pyramid down applied frameally
PARAMETERS:
frame - original image to apply pyramids to
iterations - number of times pyrUp and pyrDown should occur each
'''
def apply_pyramids(frame, iterations):

    pyrFrame = frame
    
    for j in range(iterations):
        pyrFrame = cv2.pyrDown(pyrFrame)

    for i in range(iterations):
        pyrFrame = cv2.pyrUp(pyrFrame)
     
    return pyrFrame
    
 
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
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
    contour_frame = frame.copy()

    count = 0
    for c in contours:
        if cv2.contourArea(c) > CONTOUR_THRESHOLD:
            hull = cv2.convexHull(c)
            count += 1
            # saves.append(hull)
            cv2.drawContours(contour_frame, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
    
    return None, contour_frame

def contour_mask(contour, color):
    
    r1, g1, b1 = color # Original value
    r2, g2, b2 = 0, 0, 0 # Value that we want to replace it with

    red, green, blue = contour[:,:,0], contour[:,:,1], contour[:,:,2]
    bleh = ~ ((red == r1) | (green == g1) | (blue == b1))
    contour[:,:,:3][bleh] = [r2, g2, b2]
    # contour = contour.astype('uint8')

    temp = cv2.cvtColor(contour, cv2.COLOR_RGB2GRAY)
    _, fgmask = cv2.threshold(temp, 1, 255, cv2.THRESH_BINARY)
    
    
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    return fgmask
    
def feature_detection(initial_frame, frame):
    
    return None, None

def optic_flow(frame, old_frame, mask, p0):
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    img = frame
    # If the mask is not empty
    if not (np.array_equal(np.empty(mask.shape), mask)):
        
        frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
        old_gray = cv2.cvtColor(old_frame,
                              cv2.COLOR_BGR2GRAY)
        
        if len(p0) <= 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None,
                            **feature_params)
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                            frame_gray,
                                            p0, None,
                                            **lk_params)
        
        # Select good points
        try:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        except Exception as e:
            pass
        
        
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, 
                                        good_old)):
            a, b = new.ravel()
            f, d = old.ravel()
           
            draw_mask = cv2.line(np.zeros_like(old_frame), (int(a), int(b)), (int(f), int(d)),
                            color[i].tolist(), 2)
            
            frame = cv2.circle(frame, (int(a), int(b)), 5,
                            color[i].tolist(), -1)
        # pdb.set_trace()
        img = cv2.add(frame, draw_mask)

    return img.astype(np.uint8), p0


        # gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # p0 = cv2.goodFeaturesToTrack(gray, mask = gray_mask,
        #                     **feature_params)

        # # calculate optical flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
        #                             gray,
        #                             p0, None,
        #                             **lk_params)

    
        # # Select good points
        # try:
        #     good_new = p1[st == 1]
        #     good_old = p0[st == 1]
        # except Exception as e:
        #     pass
        
        
        # # draw the tracks
        # for i, (new, old) in enumerate(zip(good_new, 
        #                                 good_old)):
        #     a, b = new.ravel()
        #     f, d = old.ravel()
        #     # pdb.set_trace()
        #     gray_mask = cv2.line(gray_mask, (int(a), int(b)), (int(f), int(d)),
        #                     color[i].tolist(), 2)
            
        #     gray = cv2.circle(gray, (int(a), int(b)), 5,
        #                     color[i].tolist(), -1)
        # #pdb.set_trace()
        # # pdb.set_trace()
        # img1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # img2 = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2RGB)
        # flow_img = cv2.add(img1, img2)

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




if __name__ == "__main__":
    main()