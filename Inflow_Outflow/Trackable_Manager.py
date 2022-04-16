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
        # print(len(temp))
        

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
    

    # Determines if a new frame with a new trackable is the same trackable present in the previous frame.
    def __trackable_already_exists(self, trackable_to_point, trackable_to_cont):

        center = trackable_to_point.get_center_point()

        contour = trackable_to_cont.get_contour_points()

        # Is center in the contour area?
        result = cv2.pointPolygonTest(contour, center, False)

        if result == 1:
            return True

        return False

    def __absorb_trackable(self, persistent_trackable, invalid_trackable):
        
        invalid_trackable.disable()
        new_conts = invalid_trackable.get_contour_points()
        persistent_trackable.add_contour(new_conts)

        return persistent_trackable


    # Determine which trackables should stay in the manager
    def __validate_trackables(self, new_trackables):
        
        old_trackables = self.old_trackables
        valid_trackables = []

        # If we have no previous trackables, we will never enter for loop. All new tracks are valid
        if old_trackables == []:
            return new_trackables

        '''
        ALERT
        - FIX THIS BUG. len(old_trackables) increases by a factor of 2
        '''
        print(len(old_trackables))

        # If we find any trackables we deem to be the same as the one in the 
        # previous frame, delete those trackables but assign their contours to a previous trackable
        for old_track in old_trackables:
            for new_track in new_trackables:
                
                # This new_track is not novel, keep the old_track and update it with new_track info
                if self.__intersection_present(old_track, new_track):
                    # Extract data from this trackable and disable it
                    self.__absorb_trackable(old_track, new_track)
                    # Keep the old_track valid, as it has not disappeared
                    valid_trackables.append(old_track)
                # This new_track is novel, make it valid
                else:
                    valid_trackables.append(new_track)

        return valid_trackables
