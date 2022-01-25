import pdb
import numpy as np
import cv2
import os
import time

main_path = os.path.dirname(os.path.abspath(__file__)) 
save_path = os.path.join(main_path, "data")
addr = os.path.join(save_path, "test2.mp4")
cap = cv2.VideoCapture(addr)
  
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

# Parameters for farenback optical flow
fb_params = [0.5, 20, 15, 3, 5, 1.2, 1]
  
def orig_kanade():
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame,
                            cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                                **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    return old_gray, p0, mask 

def draw(frame, mask, good_new, good_old):
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, 
                                    good_old)):
        a, b = new.ravel()
        f, d = old.ravel()
        #pdb.set_trace()
        mask = cv2.line(mask, (int(a), int(b)), (int(f), int(d)),
                        color[i].tolist(), 2)
        
        frame = cv2.circle(frame, (int(a), int(b)), 5,
                        color[i].tolist(), -1)
        
    img = cv2.add(frame, mask)
    return img 

def kanade(frame, mask, old_gray, p0):
    frame_gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                        frame_gray,
                                        p0, None,
                                        **lk_params)
    # Select good points
    try:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    except print(0):
        pass

    # draw the tracks 
    img = draw(frame, mask, good_new, good_old)

    # Updating Previous frame and points 
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    return img, old_gray, p0

def orig_farenback():
    # Read the video and first frame
    #cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    frame_rate = 5
    prev = 0
    return old_frame, hsv, frame_rate, prev

def farenback(new_frame, hsv, prev, frame_rate, old_frame):
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        # Do something with your image here.
        frame_copy = new_frame

        # Preprocessing for exact method
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = cv2.calcOpticalFlowFarneback(old_frame, new_frame, None, *fb_params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)


        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        old_frame = new_frame 
    
    return frame_copy, bgr, old_frame

    
## KANADE
# old_gray, p0, mask = orig_kanade()

## FARENBACK
old_frame, hsv, frame_rate, prev = orig_farenback()

  
while True:
    time.sleep(.01) # only for FARENBACK 

    # Read the next frame
    ret, frame = cap.read()

    if ret: 
        
        ## KANADE 
        # img, old_gray, p0 = kanade(frame, mask, old_gray, p0)
        # cv2.imshow('frame', img)

        ## FARENBACK
        frame_copy, bgr, old_frame = farenback(frame, hsv, prev, frame_rate, old_frame)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        
        
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break 
  
cv2.destroyAllWindows()
cap.release()