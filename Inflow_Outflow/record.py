
import os
import numpy as np
import imageio


# Modify array of frames
def start_recording(img, frames):
    
    # Convert image into np array
    frame = np.asarray(img)
    frames.append(frame)
    print("recording for {} frames".format(len(frames)))
    return frames

# Save frames to mp4 file
def save_recording(frames, save_path, file_name):
    #global keys_clicked
    
    filenames = []
    for filename in os.listdir(save_path):
        filenames += [filename]
        
    
    i = 0
    while True:
        
        file_format = '{}{}.mp4'.format(file_name, i)
        if file_format not in filenames:
            imageio.mimwrite(os.path.join(save_path, file_format), frames , fps = 2)
            print("Saving recording...")
            print("Recording saved as '{}'".format(file_format))
            frames = []
            #keys_clicked.remove('s')
            return frames
            
        i += 1
