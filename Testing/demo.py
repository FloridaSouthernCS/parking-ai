
import default
import os
import cv2
import time
import numpy as np
import denseOpticFlow
import pdb
import math
import backsub_w_countour
import default

keys_clicked = []
valid_keys = []

main_path = os.path.dirname(os.path.abspath(__file__)) 
grab_path = os.path.join(main_path, "Preprocess\\Inflow\\{}")

def main():
  type_of_data = ['Car', 'Not_Car', 'Combo', 'Demo']
  data_ind = 3

  frames = []

  files = os.listdir( 
    # Change type of data to see other cases
    grab_path.format( type_of_data[data_ind] ) 
    )
  
  for file in files:
    background_object = cv2.createBackgroundSubtractorMOG2(varThreshold=500, detectShadows=True)
    print("Filename: ", file)
    cap = cv2.VideoCapture(grab_path.format(type_of_data[data_ind] + "\\{}".format(file)))
    ret_a, frame_a = cap.read()
    
    prevgray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    ret_a, frame_a = cap.read()
    gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    while ret_a:

      
      # time.sleep(.05)
      gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
      
      flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 70, 30, 3, 5, 1.2, 0)
      
      frame_c = denseOpticFlow.draw_flow(gray, flow, 50)
      frame_d = denseOpticFlow.draw_hsv(flow)

      fgmask = backsub_w_countour.get_fgmask(background_object, frame_a)
      
      frame_e = cv2.bitwise_and(gray, gray, mask=fgmask)
      frame_f =  backsub_w_countour.get_contours(gray, fgmask)
      frame_e = cv2.cvtColor(frame_e, cv2.COLOR_GRAY2BGR)
      frame_f = cv2.cvtColor(frame_f, cv2.COLOR_GRAY2BGR)
      
      
      window1 = np.hstack([frame_a, frame_c, frame_d])
      window2 = np.hstack([frame_a, frame_e, frame_f])
      window = np.vstack([window1, window2])
      percent = .8
      window = cv2.resize(window, dsize=( math.floor(window.shape[1]*percent), math.floor(window.shape[0]*percent)) )
      #default.show_img(window, "Undetermined")
      prevgray = gray
      ret_a, frame_a = cap.read()
      
      default.start_recording(window, frames)
  default.save_recording(frames)
      


if __name__ == "__main__":
  main()



