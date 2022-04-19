'''
Trackable.py
    Make objects which can have their position saved across the 
    duration of a video feed.
'''
import numpy as np
import random
import cv2
import pdb
class Trackable:
    def __init__(self, contour, frame_shape, id):
        # A list of all the contours across the lifespan of the object. 
        self.life_contours = [contour]
        self.color = self.__random_color()
        # Most recent frame from feed, used to draw
        self.frame_shape = frame_shape
        self.enabled = True
        self.is_car = False
        self.id = id
    
    def __str__(self):
        center = self.get_center_point()
        return "ID:" + str(self.id) + " CENTER:" + str(center)
    def __repr__(self):
      return self.__str__()
    
    '''
    Getters
    '''
    def get_enabled(self):
        return self.enabled
    def get_contour_size(self, index=-1):
        return cv2.contourArea(self.life_contours[index])
    def get_func_contour_size(self, start_ind=0, end_ind=-1, step=1, function=np.mean):
        if end_ind == -1:
            end_ind = len(self.life_contours)
        sizes = np.array([])
        for i in range(start_ind, end_ind, step):
            sizes = np.append(sizes, self.get_contour_size(i))
        mean_size = function(sizes)
        
        return mean_size
        
    
    
    def get_own_bimask(self, index=-1):
        # Get the contour at some point in its lifespan
        contour = self.life_contours[index]
        # Get an empty canvas to draw contours to
        draw_frame = self.__get_zerosframe()
        # Draw the contours to our frame
        cv2.drawContours(draw_frame, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        # Make draw_frame gray
        gray_frame = self.__get_grayframe(draw_frame)
        # Make the frame a binary mask
        _, bimask = cv2.threshold(gray_frame, 10, 250, cv2.THRESH_BINARY)
        return bimask
    def get_contour_points(self, index=-1):
        return self.life_contours[index].copy()
    def get_life_contours(self):
        return self.life_contours
    def append_contour(self, contour):
        self.life_contours += [contour]
    def append_contours(self, contours):
        self.life_contours += contours
    def insert_contours(self, contours):
        self.life_contours = contours + self.life_contours
    def get_life_func(self, function=None):
        if function == None:
            function = self.get_center_point
        points = []
        for i in range(len(self.life_contours)):
            points.append(function(i))
        return points
    def get_LRTB_contour_points(self, index=-1):
        
        points = max((self.life_contours[index], ), key=cv2.contourArea)
        # get top, bottom, left, right points
        
        left_point = tuple(points[points[:, :, 0].argmin()][0])
        right_point = tuple(points[points[:, :, 0].argmax()][0])
        top_point = tuple(points[points[:, :, 1].argmin()][0])
        bottom_point = tuple(points[points[:, :, 1].argmax()][0])
        return np.array([left_point, right_point, top_point, bottom_point])
    def get_id(self):
        return self.id
    def get_left_point(self, index=-1):
        
        points = self.get_LRTB_contour_points(index)
        
        return points[1]
    
    def get_center_point(self, index=-1):
        M = cv2.moments(self.life_contours[index])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    
    def get_color(self):
        return self.color
    '''
    Setters
    '''
    
    def set_color(self, color):
        self.color = color
    def set_frame(self, frame):
        self.frame = frame
    
    def add_contour(self, contour):
        self.life_contours += [contour]
    def disable(self):
        self.enabled = False
    '''
    Private
    '''
    def __random_color(self):
        color = (
            random.randint(50, 200), 
            random.randint(50, 200), 
            random.randint(50, 200)
            )
        return color
    def __get_zerosframe(self):
        filler = np.zeros(self.frame_shape, dtype=np.uint8 )
        return filler
    def __get_grayframe(self, frame):
        return cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)