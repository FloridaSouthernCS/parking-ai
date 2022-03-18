
from collections import deque
import numpy as np


class median_frame():

  def __init__(self, max_frames=None, frames_list=None):
    if max_frames:
      self.frame_queue = deque(maxlen=max_frames)
    elif frames_list:
      self.frame_queue = deque(frames_list)
    else:
      raise Exception('Fulfil the max_frames or the frames_list parameter but not both')
  
  def add_frame(self, frame):
    self.frame_queue.append(frame)

  def get_median(self):
    frames = np.asarray(list(self.frame_queue))
    median_img = np.median(np.asarray(frames, dtype = np.uint8), axis=0).astype(np.uint8)
    return median_img
    