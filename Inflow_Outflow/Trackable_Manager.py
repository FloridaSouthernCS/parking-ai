'''
Trackable_Manager.py
    Manage Trackable objects by getting visualizations of output from them.
'''

from Trackable import Trackable
import numpy as np
import cv2
import pdb

class Trackable_Manager:

    def __init__(self, frame):
        self.id = 0
        self.old_trackables = []
        self.new_trackables = []
        self.frame = frame

    '''
    Getters
    '''
    def get_centers(self):
        centers = []
        for trackable in self.new_trackables:
            centers.append(trackable.get_center_point())
        return centers
            
    def get_contours(self):
        contours = []
        for trackable in self.new_trackables:
            contours.append(trackable.get_contour())
        return contours

    # Returns a frame of every trackable colorized
    def get_trackable_frame(self):
        draw_frame = self.frame.copy()
        for trackable in self.new_trackables:
            contour = trackable.get_contour_points()
            # If this trackable is not disabled
            if trackable.enabled:
                cv2.drawContours(draw_frame, [contour], -1, trackable.get_color(), thickness=cv2.FILLED)
        
        return draw_frame

    def get_frame_shape(self):
        return np.asarray(self.frame).shape

    def get_trackables_from_contours(self, contours):
        trackables = []
        for c in contours:
            trackables.append(Trackable(c, self.get_frame_shape(), self.id))
            self.id += 1
        
        return trackables

    '''
    Setters
    '''

    def set_frame(self, frame):
        self.frame = frame

    def add_trackables(self, trackables):
        
        self.old_trackables = self.new_trackables.copy()
        
        # The new trackables are deleted if they are determined to be an old_trackable that moved.
        # Else, they are added to new_trackables
        
        temp = self.__validate_trackables(trackables)
        self.new_trackables = temp
        

    '''
    Private
    '''
    # Takes in 2 arrays of contours 
    def __intersection_present(self, trackable1, trackable2):
        contour1 = trackable1.get_contour_points()
        contour2 = trackable2.get_contour_points()
        # Two separate contours trying to check intersection on
        contours = [contour1, contour2]
        
        # Create image filled with zeros the same size of original image
        blank = np.zeros(self.get_frame_shape()[0:2])

        # Copy each contour into its own image and fill it with '1'
        image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
        image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

        # Use the logical AND operation on the two images
        # Since the two images had bitwise and applied to it,
        # there should be a '1' or 'True' where there was intersection
        # and a '0' or 'False' where it didnt intersect
        intersection = np.logical_and(image1, image2)

        # Check if there was a '1' in the intersection
        return intersection.any()
    

    # Checks if the center of one trackable is in the contour of a different trackable
    def __center_in_contour(self, trackable_to_point, trackable_to_cont):

        center = trackable_to_point.get_center_point()

        contour = trackable_to_cont.get_contour_points()

        # Is center in the contour area?
        result = cv2.pointPolygonTest(contour, center, False)

        if result == 1:
            return True

        return False

    # Used to transfer data from an invalid trackable to a persistent trackable
    def __absorb_trackable(self, persistent_trackable, invalid_trackable):
        
        invalid_trackable.disable()
        new_conts = invalid_trackable.get_contour_points()
        persistent_trackable.add_contour(new_conts)

        return persistent_trackable


    # Decides whether a trackable is persistent from a previous frame or brand new. This function removes trackables which have not been seen in a previous frame.
    # Returns list of trackables that have either been updated or are new.
    def __validate_trackables(self, new_trackables):
        
        old_trackables = self.old_trackables

        # Modifies the new_trackables array by removing new_trackables that are actually old_trackables that moved.
        for i in range(len(old_trackables)):
            old_track = old_trackables[i]
            for j in range(len(new_trackables)):
                new_track = new_trackables[j]
                # If the center of old is in the contour of new, update the old_trackable and add it to the return array
                if self.__center_in_contour(old_track, new_track):
                    old_track = self.__absorb_trackable(old_track, new_track)
                    new_trackables[j] = old_track

        return new_trackables
