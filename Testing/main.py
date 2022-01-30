
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

    # Track what keys are pressed
    keys_clicked = []
    _thread.start_new_thread(check_key, (keys_clicked,))

    # Default Garden enterance IP
    addr = 'http://10.7.0.19/image4?res=half&quality=1&doublescan=0'
    
    # pull_from_web(addr, keys_clicked)

    # Saved mp4 examples 
    #addr = os.path.join(save_path, "test2.mp4")
    #pull_from_addr(addr, keys_clicked)
    pull_from_web(addr, keys_clicked=keys_clicked)
    #ret, frame = vid.read()

# Pulls saved feed
def pull_from_addr(addr, keys_clicked):
    pdb.set_trace()
    # pull video from addr filepatth
    try:
        vid = cv2.VideoCapture(addr)
    except Exception as e:
        print(e)

    
    # while q is not hit
    while 'q' not in keys_clicked:

        # grab frame
        try:
            ret, frame = vid.read()
        except Exception as e:
            print(e)
            continue
        
        
        # show frame 
        show_img(frame)

    
    vid.release()
    cv2.destroyAllWindows()

# Used to watch live video from web
def pull_from_web(addr, keys_clicked):
    print("==== KEY COMMANDS ====")
    print(" 'r' = start/stop record ")
    print(" 'q' = exit program ")
    frames = []
    
    # Continue until quit occurs
    while 'q' not in keys_clicked:
        
        # Pull image from addr
        feed = request_img(addr)

        # Convert image into array
        img = np.asarray(Image.open(BytesIO(feed.content)))

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Show image in window
        show_img(gray_img)
    
        #Record if 'r' has been toggled
        start_recording(gray_img, frames, keys_clicked)

    # If there were frames pulled, save a recording
    save_recording(frames)

# Modify array of frames
def start_recording(img, frames, keys_clicked):
    if ('r' in keys_clicked and (keys_clicked.count('r')/2) % 2 > 0):
        # Convert image into np array
        frame = np.asarray(img)
        frames.append(frame)
        print("recording")
    return frames

# Save frames to mp4 file
def save_recording(frames):
    if frames != []:
        imageio.mimwrite(os.path.join(save_path,'test17.mp4'), frames , fps = 2)
        print("Recording saved as '{}'".format('test17.mp4'))

# Display the image
def show_img(img):
    window = cv2.resize(img, (600,400) )
    cv2.imshow('Video', window)
    cv2.waitKey(1)

# Request image from web
def request_img(request, verbose=0):
    if verbose == 1: print("Grabbing frame...")
    start = time.time()
    feed = requests.get(request)
    end = time.time()
    if verbose == 1: print("Frame aquired in", end-start, "seconds.\n")
    return feed

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