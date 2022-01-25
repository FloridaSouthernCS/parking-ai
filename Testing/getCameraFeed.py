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
import matplotlib.pyplot as plt

main_path = os.path.dirname(os.path.abspath(__file__)) 
save_path = os.path.join(main_path, "preprocess")


def main():
    
    # Default Garden enterance IP
    #addr = 'http://10.7.0.19/image4?res=half&quality=1&doublescan=0'
    addr = os.path.join(save_path, "test2.mp4")
    print("==== KEY COMMANDS ====")
    print(" 'r' = start/stop record ")
    print(" 'q' = exit program ")

    # Track what keys are pressed
    keys_clicked = []
    _thread.start_new_thread(check_key, (keys_clicked,))

    pull_feed(addr, keys_clicked)

    
def pull_feed(addr, keys_clicked):
    frames = []
    count = 0
    videofeed = cv2.VideoCapture(addr)
    # Continue until quit occurs
    while 'q' not in keys_clicked:
        # Pull image from camera
        
        # feed, spf  = request_img(addr)
        ret, feed = videofeed.read()

        # Convert image into array
        #img = np.asarray(Image.open(BytesIO(feed.content)))
        img = feed
        
        show_img(img)
        # Show image in window
        #window = imutils.resize(img, width=400)
        
    videofeed.release()
    cv2.destroyAllWindows()
        # Record if 'r' has been toggled
    #     if ('r' in keys_clicked and (keys_clicked.count('r')/2) % 2 > 0):

    #         print('recording...', count)

    #         # Convert image into np array
    #         frame = np.asarray(img)
    #         frames.append(frame)
      
    #     count += 1

    # # If there were frames pulled, save a recording
    # if frames != []:
    #     imageio.mimwrite(os.path.join(save_path,'test.mp4'), frames , fps = 2)
    

def show_img(img):
    window = cv2.resize(img, (600,400) )
    cv2.imshow('Video', window )
    cv2.waitKey(1)

def request_img(request, verbose=0):
    if verbose == 1: print("Grabbing frame...")
    start = time.time()
    feed = requests.get(request)
    end = time.time()
    if verbose == 1: print("Frame aquired in", end-start, "seconds.\n")
    return feed, end-start


# Check if a key was clicked
def check_key(a_list):
    while True:
        try:
            key = keyboard.read_key()
            a_list.append(key)
        except:
            pass



if __name__ == "__main__":
    main()