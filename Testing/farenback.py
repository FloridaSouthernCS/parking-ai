import pdb
import time
import numpy as np
import cv2
import os

main_path = os.path.dirname(os.path.abspath(__file__)) 
grab_path = os.path.join(main_path, "postprocess2")
addr = os.path.join(grab_path, "test4.mp4")
cap = cv2.VideoCapture(addr)


def dense_optical_flow(method, video_path, params=[], to_gray=True):
    # Read the video and first frame
    #cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_rate = 5
    prev = 0
    while True:
        
        time.sleep(.01)
        # Read the next frame
        ret, new_frame = cap.read()
        

        time_elapsed = time.time() - prev
        res, image = cap.read()

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            # Do something with your image here.
            frame_copy = new_frame
            if not ret:
                break

            # Preprocessing for exact method
            if to_gray:
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            flow = method(old_frame, new_frame, None, *params)

            # Encoding: convert the algorithm's output into Polar coordinates
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Use Hue and Value to encode the Optical Flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            # Convert HSV image into BGR for demo
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("frame", frame_copy)
            cv2.imshow("optical flow", bgr)
            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                break

            # Update the previous frame
            old_frame = new_frame




def main():
    dense_optical_flow(cv2.calcOpticalFlowFarneback, 'sample3.mp4',[0.5, 10, 10, 3, 10, 1.2, 0])
    cv2.calcOpticalFlowFarneback()

main()