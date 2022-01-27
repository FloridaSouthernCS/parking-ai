from __future__ import print_function
import time
import cv2 as cv
import argparse
import os
import numpy as np
import imageio
import scipy.ndimage as sp


# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()

main_path = os.path.dirname(os.path.abspath(__file__)) 
grab_path = os.path.join(main_path, "preprocess")
addr = os.path.join(grab_path, "test2.mp4")
save_path = os.path.join(main_path, "postprocess")


# 2000 - 7 / 900 - 9
backSub_knn = cv.createBackgroundSubtractorKNN(dist2Threshold=900, detectShadows=False)

# 500 - 7 / 900 - 9
backSub_mog = cv.createBackgroundSubtractorMOG2(varThreshold= 500, detectShadows=False)
capture = cv.VideoCapture(addr)
frames = []
def main():
    while True:
        # time.sleep(.1)
        ret, frame = capture.read()
        if frame is None:
            break

        frame = sp.gaussian_filter(frame, sigma = 7)


        # fgMask_knn = backSub_knn.apply(frame)
        # frame_sub = fgMask_knn 

        fgMask_mog = backSub_mog.apply(frame)
        
        fgMask_knn = backSub_knn.apply(fgMask_mog)
        
        frame_sub = fgMask_mog



        
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', frame_sub)
        
        
        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break
        start_recording(frame_sub, frames)


    #save_recording(frames)

def color_threshold():
    pass

# Modify array of frames
def start_recording(img, frames):
    
    # Convert image into np array
    frame = np.asarray(img)
    frames.append(frame)
    return frames

# Save frames to mp4 file
def save_recording(frames):
    
    imageio.mimwrite(os.path.join(save_path,'test1.mp4'), frames , fps = 2)
    print("Recording saved as '{}'".format('test1.mp4'))


main()