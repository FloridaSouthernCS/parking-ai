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

main_path = os.path.dirname(os.path.abspath(__file__)) 
save_path = os.path.join(main_path, "data")


def main():
    frames = []
    count = 0

    while count != 50: 
        frame = requestImg()
        img = Image.open(BytesIO(frame.content))
        frame = np.asarray(img)
        frames.append(frame)
      
        count+= 1
    imageio.mimwrite(os.path.join(save_path,'test.mp4'), frames , fps = 2)
    


def requestImg(request='http://10.7.0.19/image4?res=half&quality=1&doublescan=0'):
    print("Grabbing frame...")
    start = time.time()
    frame = requests.get(request)
    end = time.time()
    print("Frame aquired in", end-start, "seconds.\n")
    return frame



if __name__ == "__main__":
    main()