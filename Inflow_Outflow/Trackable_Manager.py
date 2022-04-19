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
        self.retired_trackables = []
    '''
    Getters
    '''
    def get_centers(self):
        centers = []
        for trackable in self.new_trackables:
            centers.append(trackable.get_center_point())
        return centers
    def get_retired_trackables(self):
        return self.retired_trackables
            
    def get_contours(self):
        contours = []
        for trackable in self.new_trackables:
            contours.append(trackable.get_contour())
        return contours
    # Returns a frame of every trackable colorized
    def get_trackable_contours_frame(self):
        draw_frame = self.frame.copy()
        for trackable in self.new_trackables:
            contour = trackable.get_contour_points()
            
            cv2.drawContours(draw_frame, [contour], -1, trackable.get_color(), thickness=5)
            
        return draw_frame
    # Get a frame which displays a 'flow' of traceable points
    def get_traced_frame(self, function=Trackable.get_center_point):
        draw_frame = self.frame.copy()
        for trackable in self.new_trackables:
            color = trackable.get_color()
            trace_point1 = function(trackable, 0)
            # Draw a line from position n to n+1, then n = n+1
            for i in range(1, len(trackable.get_life_contours())):
                
                trace_point2 = function(trackable, i)
                cv2.line(draw_frame, trace_point1, trace_point2, color, 5)
                trace_point1 = function(trackable, i)
            # Draw a circle at the most recent index for a traceable point
            cv2.putText(draw_frame, str(trackable.get_id()), function(trackable), cv2.FONT_HERSHEY_SIMPLEX, 2, color=color, thickness=8)
            cv2.circle(draw_frame, function(trackable), 10, color, -1)
        return draw_frame
    # Returns a list of the left, right, top, and bottom-most points for each Trackable
    def get_extreme_points(self):
        points = []
        for track in self.new_trackables:
            points.append(track.get_LRTB_points())
        return points
    def get_frame_shape(self):
        return np.asarray(self.frame).shape
    # Creates new trackable objects for every contour that is present
    def generate_trackables(self, contours):
        trackables = []
        for c in contours:
            trackables.append(Trackable(c, self.get_frame_shape(), self.id))
            self.id += 1
        
        return trackables
    # Returns a list of all current trackables in the most recent frame
    def get_trackables(self):
        return self.new_trackables
    '''
    Setters
    '''
    # Set the frame that all trackables will be drawn onto
    def set_frame(self, frame):
        self.frame = frame
    # Propose a set of contours as candidates for new trackables. Returns a list the trackables it deems valid
    ''' SAVE_RETIRED_TRACKABLES SHOULD ONLY BE SET TO TRUE IF DATA IS CURRENTLY BEING LABELED BY HUMANS.'''
    def propose_trackables(self, trackables, save_retired=False):
        self.old_trackables = self.new_trackables.copy()
        # The new trackables are deleted if they are determined to be an old_trackable that moved.
        # Else, they are added to new_trackables
        self.new_trackables = self.__validate_trackables(trackables, save_retired)
    def retire_all_trackables(self):
        self.retired_trackables += self.new_trackables        
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
        new_conts = invalid_trackable.get_life_contours()
        persistent_trackable.append_contour(new_conts)
        
        return persistent_trackable
    # Decides whether a trackable is persistent from a previous frame or brand new. This function removes trackables which have not been seen in a previous frame.
    # Returns list of trackables that have either been updated or are new.
    def __validate_trackables(self, new_trackables, save_retired=False):
        
        old_trackables = self.old_trackables
        
        # Modifies the new_trackables array by removing new_trackables that are actually old_trackables that moved.
        for i in range(len(old_trackables)):
            old_track = old_trackables[i]
            absorb = False
            for j in range(len(new_trackables)):
                new_track = new_trackables[j]
                # If the center of old is in the contour of new, update the old_trackable and add it to the return array
                if self.__center_in_contour(old_track, new_track):
                    old_track = self.__absorb_trackable(old_track, new_track)
                    new_trackables[j] = old_track
                    absorb = True
                
                    
            # If we know this old_track did not inherit a new_trackable, it will be retired so save it
            if not absorb and save_retired: self.retired_trackables += [old_track]
        return new_trackables