import cv2
import time
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os 
import pdb
import imageio
import keyboard
import _thread
import imutils

main_path = os.path.dirname(os.path.abspath(__file__)) 
save_path = os.path.join(main_path, "data")


def main():
    print("==== KEY COMMANDS ====")
    print(" 'q' = stop pulling feed ")
    print(" 'r' = start/stop record ")
    print()

    # Track what keys are pressed
    a_list = []
    _thread.start_new_thread(input_thread, (a_list,))

    frames = []
    count = 0

    while 'q' not in a_list: 
        # Pull image from camera
        feed = requestImg()

        # Convert image into array
        img = np.asarray(Image.open(BytesIO(feed.content)))
        window = imutils.resize(img, width=400)
        cv2.imshow('Live Camera Feed', window )
        cv2.waitKey(1)

        # Record if 'r' has been toggled
        if ('r' in a_list and (a_list.count('r')/2) % 2 > 0):

            print('recording...', count)

            # Convert image into np array
            frame = np.asarray(img)
            frames.append(frame)
      
        count += 1

    # If there were frames pulled, save a recording
    if frames != []:
        imageio.mimwrite(os.path.join(save_path,'test.mp4'), frames , fps = 2)
    


def requestImg(request='http://10.7.0.19/image4?res=half&quality=1&doublescan=0'):
    print("Grabbing frame...")
    start = time.time()
    feed = requests.get(request)
    end = time.time()
    print("Frame aquired in", end-start, "seconds.\n")
    return feed


def input_thread(a_list):
    while True:
        
        try:
            
            key = keyboard.read_key()
            a_list.append(key)
            
        except:
            pass
        
    
# def main():
#     a_list = []
#     _thread.start_new_thread(input_thread, (a_list,))
#     while 'q' not in  a_list:
#         open_feed()



if __name__ == "__main__":
    main()