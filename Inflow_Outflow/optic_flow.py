

import cv2
import numpy as np
import pdb

class lk_optic_flow:

    def __init__(self, first_frame, feature_params, lk_params, p0=None, mask=None):
        self.old_frame = None
        self.new_frame = first_frame
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.p0 = p0
        if type(mask) == type(None): mask = np.zeros(first_frame.shape)
        self.mask = mask

    def get_flow(self, frame):
        self.old_frame = self.new_frame
        self.new_frame = frame
        flow_frame = self.__lk_flow()
        return flow_frame

    def set_mask(self, mask):
        self.mask = mask

    def set_p0(self, p0):
        self.p0 = p0

    def __reset_p0(self):
        frame_gray = self.__flip_channels( self.new_frame )
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None,
                            **self.feature_params)
        self.p0 = p0
        return p0

    def __flip_channels(self, frame):
        frame = frame.astype(np.uint8)
        if len(frame.shape) > 2: 
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
  		
    def __lk_flow(self):
        frame = self.new_frame
        old_frame = self.old_frame

        color = np.random.randint(0, 255, (100, 3))
        mask = self.__flip_channels(self.mask)

        # Default img initialization for return statement
        img = frame
        
        # If the mask is not empty
        if not (np.array_equal(np.empty(mask.shape), mask)):
            
            frame_gray = self.__flip_channels(frame)
            old_gray = self.__flip_channels(old_frame)
            
            # if len(p0) == 0:
            #     print('t')
            #     p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None,
            #                     **self.feature_params)
            p0 = self.__reset_p0()
        
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                frame_gray,
                                                p0, None,
                                                **self.lk_params)
            
            # Select good points
            try:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            except Exception as e:
                pass
            
            
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, 
                                            good_old)):
                a, b = new.ravel()
                f, d = old.ravel()
            
                draw_mask = cv2.line(np.zeros_like(old_frame), (int(a), int(b)), (int(f), int(d)),
                                color[i].tolist(), 2)
                
                frame = cv2.circle(frame, (int(a), int(b)), 5,
                                color[i].tolist(), -1)
            # pdb.set_trace()
            img = cv2.add(frame, draw_mask).astype(np.uint8)
        
        return img






