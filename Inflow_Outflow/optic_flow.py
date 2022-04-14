

import cv2
import numpy as np
import pdb
import random


class lk_optic_flow:
  

    def __init__(self, first_frame, feature_params, lk_params, p0=[], mask=None):
        self.old_frame = None
        self.new_frame = first_frame
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.p0 = p0
        if type(mask) == type(None): mask = np.zeros(first_frame.shape)
        self.mask = mask

    def get_flow(self, frame, points):
        self.old_frame = self.new_frame
        self.new_frame = frame
        flow_frame = self.__point_tracking(points)
        return flow_frame

    def set_mask(self, mask):
        self.mask = mask

    def set_p0(self, p0):
        self.p0 = p0

    def __flip_channels(self, frame):
        frame = frame.astype(np.uint8)
        if len(frame.shape) > 2: 
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    def __random_color(self):
        r = random.randint(0,150)
        g = random.randint(0,150)
        b = random.randint(0,150)
        return (r, g, b)

    def __point_tracking(self, points):
        p0 = self.p0
        frame = self.new_frame
        old_frame = self.old_frame

        color = np.random.randint(0, 255, (100, 3))
        mask = self.__flip_channels(self.mask)

        # Default img initialization for return statement
        img = frame
        
        
        # If the mask is not empty
        if not (np.array_equal(np.empty(mask.shape), mask)) and p0 != []:
            
            p1 = points
            draw_mask = np.zeros_like(old_frame)
            # draw the tracks
            for i, (new, old) in enumerate(zip(p1, 
                                            p0)):
                color = self.__random_color()
                draw_mask = cv2.line(draw_mask, (new[0], new[1]), (old[0], old[1]),
                                color, 5)
                
                frame = cv2.circle(frame, (new[0], new[1]), 10,
                                color, -1)

                # pdb.set_trace()

                img = cv2.add(frame, draw_mask).astype(np.uint8)
            self.p0 = p1
        else:
            self.p0 = points
        
        return img





