

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
  		
    def __lk_flow(self, points):
        frame = self.new_frame
        old_frame = self.old_frame

        color = np.random.randint(0, 255, (100, 3))
        mask = self.__flip_channels(self.mask)

        # Default img initialization for return statement
        img = frame
        
        # pdb.set_trace()
        # If the mask is not empty
        if not (np.array_equal(np.empty(mask.shape), mask)):
            
            frame_gray = self.__flip_channels(frame)
            old_gray = self.__flip_channels(old_frame)
            
            # if len(p0) == 0:
            #     print('t')
            # temp = cv2.goodFeaturesToTrack(frame_gray, mask = None,
            #                     **self.feature_params)

            # p0 = self.__reset_p0()
            # pdb.set_trace()

            # pdb.set_trace()
            # p0 = np.asarray([[points[0], points[1]]]).astype('float32')
            # p0 = np.asarray([[points[0], points[1]]]).astype('float32')
            self.p0 = points
            p0 = points
           

        
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0, None, **self.lk_params)
            
            # Select good points
            try:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            except Exception as e:
                pass

            # try:
            #     good_new = p1[0]
            #     good_old = p0[0]
            # except Exception as e:
            #     pass
            
            # pdb.set_trace()
            


            # a, b = p1[0][0], p1[0][1]
            # f, d = p0[0][0], p0[0][1]



            # draw_mask = cv2.line(np.zeros_like(old_frame), (int(a), int(b)), (int(f), int(d)),
            #                     (0,0,255), 7)
                
            # img = cv2.circle(frame, (int(f), int(d)), 8,
            #                 (0,0,255), -1)
            
            
            
            
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, 
                                            good_old)):
                # pdb.set_trace()
                a, b = new.ravel()
                f, d = old.ravel()
            
                # draw_mask = cv2.line(np.zeros_like(old_frame), (int(a), int(b)), (int(f), int(d)),
                #                 color[i].tolist(), 5)
                
                # frame = cv2.circle(frame, (int(a), int(b)), 5,
                #                 color[i].tolist(), -1)

                draw_mask = cv2.line(np.zeros_like(old_frame), (int(a), int(b)), (int(f), int(d)),
                                (0, 255,0), 5)
                
                frame = cv2.circle(frame, (int(a), int(b)), 5,
                                (0, 255,0), -1)
            # # pdb.set_trace()
                img = cv2.add(frame, draw_mask).astype(np.uint8)
        
        return img
    
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





