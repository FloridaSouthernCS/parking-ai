from __future__ import print_function
import cv2 as cv
import argparse
import os
import numpy as np
import imageio
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

backSub = cv.createBackgroundSubtractorMOG2(varThreshold=2000, detectShadows=False)
capture = cv.VideoCapture(addr)
frames = []
def main():
    while True:
        
        ret, frame = capture.read()
        if frame is None:
            break
        
        fgMask = backSub.apply(frame)
        
        
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)
        
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        start_recording(fgMask, frames)


    save_recording(frames)

# Modify array of frames
def start_recording(img, frames):
    
    # Convert image into np array
    frame = np.asarray(img)
    frames.append(frame)
    return frames

# Save frames to mp4 file
def save_recording(frames):
    
    imageio.mimwrite(os.path.join(save_path,'test.mp4'), frames , fps = 2)
    print("Recording saved as '{}'".format('test.mp4'))


main()