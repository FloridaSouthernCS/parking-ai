import cv2
import time
from PIL import Image
import numpy as np
import requests
from io import BytesIO

def main():
    while True:
        
        
        frame = requestImg();
        img = Image.open(BytesIO(frame.content))


        # temp = img.getdata()
        # print(np.array(temp)[0])
        # time.sleep(5)

def requestImg(request='http://10.7.0.19/image4?res=half&quality=1&doublescan=0'):
    print("Grabbing frame...")
    start = time.time()
    frame = requests.get(request)
    end = time.time()
    print("Frame aquired in", end-start, "seconds.\n")
    return frame



if __name__ == "__main__":
    main()