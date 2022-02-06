
import default
import os
import cv2
import time
import numpy as np
import denseOpticFlow
import pdb

keys_clicked = []
valid_keys = []

main_path = os.path.dirname(os.path.abspath(__file__)) 
grab_path = os.path.join(main_path, "Preprocess\\Inflow\\{}")

def main():
  type_of_data = ['Car', 'Not_Car', 'Combo']
  data_ind = 0

  files = os.listdir( 
    # Change type of data to see other cases
    grab_path.format( type_of_data[data_ind] ) 
    )[::-1]
  background_object = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=True)
  for file in files:
    
    cap = cv2.VideoCapture(grab_path.format(type_of_data[data_ind] + "\\{}".format(file)))
    ret_a, frame_a = cap.read()
    
    prevgray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    while ret_a:

      ret_a, frame_a = cap.read()
      # time.sleep(.05)
      gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
      
      flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 40, 10, 3, 5, 1.2, 0)

      frame_c = denseOpticFlow.draw_flow(gray, flow, 50)
      frame_d = denseOpticFlow.draw_hsv(flow)

      window = np.vstack([frame_a, frame_c, frame_d])
      
      default.show_img(frame_d, "Undetermined")
      
      


if __name__ == "__main__":
  main()



