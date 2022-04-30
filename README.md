# parking-ai
Detecting Cars in Parking Lots using AI (CSC 3992: Directed Study)
- Brock Wilson and Hannah Wilberding, Supervised by Dr. Eicholtz, Spring 2022

In this folder there are 8 Python files:
1. inflow_main.py - The main computer vision file. Calls functions from inflow.py
2. inflow.py - The functions called from main
3. insights.py - The main machine learning file. Uses an SVM
4. key_log.py - A simple keylogger
5. read_write.py - Formats a dataset into a text file given the arrays provided.
6. record.py - Uses keylogger to start, stop, quit, or record an output feed.
7. Trackable_Manager.py - Holds several arrays of trackable objects and manipulates the trackables in those arrays
8. Trackable.py - Called by trackable_manager to create new trackable objects. Trackable objects contain contour areas over a set of frames


Overview:
  To display the feed, go to inflow_main.py and run the main function.
  To label the data, uncomment 'read_write.label_data(track_man, addr)' at the end of the while loop in the main function and run it
  To run the support vector machine, go to insights.py and run the file. This program uses saved data from data labeling mentioned above.

Inflow_main.py displays 6 frames in a window:
* frame_norm: the input frame but normalized a bit. Visually similar to the original frame
* backsub_frame: The frame after motion-based background subtraction. This frame represents the pixel-level accuracy of what is detected as movement.
* contour_foreground: The frame which best draws lines around the background subtraction. In this case, you will notice there are no concave lines. We used a cv2.hull(contours) method to produce an entirely convex shape. We use the convex shape to essentially remove the most amount of false-negative pixels found by background subtraction. 
* track_frame: A frame displaying all of the trackables on the screen. Each area should have it's own distinct color
* traced_points_frame: A frame displaying the midpoints of each trackable for each trackable's entire lifespan. Also displays the ID.
* triange_frame: A frame displaying the reduced set of information provided by traced_points_frame. Because a trackable's set of midpoints can vary in size (due to duration of time on the video), a way to remove this variability is required to put the information into a machine learning algorithm. In this case, we reduce the variable-length data into exactly 3 numbers. The RGB triangle drawn is used to more easily represent these numbers.


Trackable_manager notes:

* Persistence functions:
  Trackable_Manager has several persistence functions. We define a new trackable as persistent if it can be evaluated to be the same contour as one in a previous frame. 

  We have several unused functions that can be used in the future for experimentation. If you were to describe whether an object in one frame is the same as the next, you might say that if there is an intersection between these objects then they are the same one just moved over. We found that this can cause issues with merging noise at the gate with an entering object. To experiment, go to the __validate_trackables function and change '__if_center_in_contour' to '__intersection_present' or some other function to determine if two contours are the same.

* Dominance functions
  Trackable_Manager also has several dominance functions. We use dominance functions to determine which trackable should be absorbed by a more significant or important trackable. 

  When a trackable absorbs another, it inherits the absorbee trackable's position and size for that single frame and discards the rest of the absorbee's lifespan of positions and sizes. 

  The goal here is to disallow noise or several contours on a vehicle from removing the history of a vehicle traversing the frame.
