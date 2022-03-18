
# import enum
import cv2
import time
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os 
import pdb
import imageio
import matplotlib.pyplot as plt
from pynput.keyboard import Listener
import median_frame_queue

main_path = os.path.dirname(os.path.abspath(__file__))
save_folder = "Preprocess/Inflow/Car"
save_path = os.path.join(main_path, save_folder)
keys_clicked = []
valid_keys = []

# main_path = os.path.dirname(os.path.abspath(__file__)) 
grab_path = os.path.join(main_path, "Preprocess\\Inflow\\Not_Car")

# addr = os.path.join(grab_path, "large_white_night.mp4")
# cap = cv2.VideoCapture(addr)


def main():
    global valid_keys
    # Track what keys are pressed
    valid_keys += ['r', 's', 'q', 'd']
    # check_key()
    
    
    # Default Garden enterance IP
    #addr = 'http://10.7.0.19/image4?res=half&quality=1&doublescan=0'
    addr = os.path.join(grab_path, "not_car10.mp4")
    
    pull_from_addr(addr, keys_clicked)
    
    
    # Saved mp4 examples 
    #addr = os.path.join(save_path, "test2.mp4")
    #pull_from_addr(addr, keys_clicked)
    #pull_from_web('http://10.7.0.17/image3?res=half&quality=1&doublescan=0', keys_clicked=keys_clicked)
    #ret, frame = vid.read()

# Pulls saved feed
def pull_from_addr(addr, keys_clicked):
    
    # pull video from addr filepatth
    try:
        vid = cv2.VideoCapture(addr)
    except Exception as e:
        print(e)
    median = median_frame_queue.median_frame(60)
    frames = []
    ret, frame = vid.read()
    # while q is not hit
    while 'q' not in keys_clicked and ret:
        
        frames.append(frame)
        median.add_frame(frame)
        # show frame 
        show_img(frame)
        
        # grab frame
        try:
            ret, frame = vid.read()
        except Exception as e:
            print(e)
            continue
        
    print(len(frames))
        

    show_img(median.get_median())
    
    vid.release()
    pdb.set_trace()
    cv2.destroyAllWindows()

def median(frames):
    frames = np.asarray(frames)
    # grays = []
    # for frame in frames:
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     grays.append(gray)
    median_img = np.median(np.asarray(frames, dtype = np.uint8), axis=0).astype(np.uint8)
    # median_img = cv2.cvtColor(median_img, cv2.COLOR_GRAY2BGR)

    return median_img

# Used to watch live video from web
def pull_from_web(addr, keys_clicked):
    print("==== KEY COMMANDS ====")
    print(" 'r' = start/stop record ")
    print(" 's' = save recording ")
    print(" 'd' = delete queued frames ")
    print(" 'q' = exit program ")
    frames = []
    
    # Continue until quit occurs
    stime = 0
    ctime = 0
    fps = 0
    while 'q' not in keys_clicked:
        
        # Get FPS
        if ctime:
            fps = format(1/(ctime-stime),".2f")

        # Get start time
        stime = time.time()

        # Pull image from addr
        feed = request_img(addr)

        # Convert image into array
        img = np.asarray(Image.open(BytesIO(feed.content)))

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Show image in window
        show_img(gray_img, fps)
    
        #Pass recording params
        frames = handle_key_recording(gray_img, frames, keys_clicked)

        # Get current time
        ctime = time.time()
            
def handle_key_recording(img, frames, keys_clicked):

    if 'd' in keys_clicked:
        keys_clicked.remove('d')
        frames = []
        print("Frames deleted")

    # Every other time r is clicked, record
    if ('r' in keys_clicked and (keys_clicked.count('r')) % 2 > 0):
        frames = start_recording(img, frames)

    # Save every time s is clicked
    if 's' in keys_clicked:
        if frames != []:
            frames = save_recording(frames)
        else:
            keys_clicked.remove('s')
            print("There are no frames to save")
    return frames

# Modify array of frames
def start_recording(img, frames):
    
    # Convert image into np array
    frame = np.asarray(img)
    frames.append(frame)
    print("recording for {} frames".format(len(frames)))
    return frames

# Save frames to mp4 file
def save_recording(frames):
    #global keys_clicked
    
    filenames = []
    for filename in os.listdir(save_path):
        filenames += [filename]
        
    
    i = 0
    while True:
        
        file_format = '{}{}.mp4'.format(save_folder, i)
        if file_format not in filenames:
            imageio.mimwrite(os.path.join(save_path, file_format), frames , fps = 2)
            print("Saving recording...")
            print("Recording saved as '{}'".format(file_format))
            frames = []
            #keys_clicked.remove('s')
            return frames
            
        i += 1

# Display the image
def show_img(img, fps=0):
    
    window = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2) )
    cv2.rectangle(window, (0,0), (80, 10), (127,127,127), -1)
    cv2.putText(window, "FPS: {}".format(fps), (0,10), cv2.FONT_HERSHEY_TRIPLEX, .5, (255, 255, 255), 1, 2)
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
def check_key():
    with Listener(
            on_press=on_press
            ) as listener:
        listener.join()
    
    
def on_press(key):
    global keys_clicked
    global valid_keys
        
    try:
        if key.char in valid_keys:
            keys_clicked.append(key)
            print(keys_clicked, " clicked")
    except print(0):
        pass


if __name__ == "__main__":
    main()