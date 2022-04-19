# pca_knn.py
import ast
from Trackable_Manager import Trackable_Manager 
from Trackable import Trackable
import pdb
import os
import numpy as np

main_path = os.path.dirname(os.path.abspath(__file__)) 
datapath = os.path.join(main_path, "Data", "Inflow")
train_labels_path = os.path.join(datapath, "training_labels.txt")
train_data_path = os.path.join(datapath, "training_data.txt")

def label_data(track_man, reference_video):
    
    labels = []
    data = []
    print()
    print("NOTE: Only label a contour as noise if it is noise in the last several frames of its lifespan")
    print("NOTE: If the video is severely littered with noise, do not continue with the labeling process here for that video")
    
    for trackable in track_man.get_retired_trackables():
        # This is only here to validate functions exist. Comment out when running and delete when done
        result = input("Trackable: " + str(trackable.get_id()) + " is a car(c), not_car(n), noise([enter]), delete(d): ")
        
        # Aquire training label
        if result == "c":
            labels.append(1)
        elif result == "n":
            labels.append(0)
        elif result == "d":
            continue
        else:
            labels.append(-1)

        # Aquire training data
        # If we don't have enough data to work with, its probably not a car
        if len(trackable.get_life_func()) >= 3:
            temp_data = [trackable.get_life_func()]
            temp_data.append(trackable.get_func_contour_size(function=np.mean))
            temp_data.append(trackable.get_func_contour_size(function=np.median))
            temp_data.append(trackable.get_func_contour_size(function=np.max))

            start = np.array(temp_data[0][0])
            middle = np.array(temp_data[0][len(temp_data[0])//2])
            end = np.array(temp_data[0][-1])
            print("start to end: ", dist(start, end))
            print("gait: ", gait(start, middle, end))
            print("acceleration: ", accel(start, middle, end))
            
        
            data.append(temp_data)
            
    
    result = input("All results are correct? Yes(y), No([enter]): ")

    # Save files
    saved1 = False
    saved2 = False
    if result == 'y':
        saved1 = save_to_file(train_labels_path, reference_video, labels )
        saved2 = save_to_file(train_data_path, reference_video, data )

    if saved1 and saved2:
        print("Files saved.")
    else:
        print("Some files did not save.")

    
def read_file(path):

    with open(path, "r") as file:
        # Writing data to a file
        info = file.read().split('\n')
        file.close()

    return info

def purge_references(data):
    new_data = []
    for item in data:
        
        if not '<' in item and item != '':
            new_data.append(item)
    
    return new_data

def str_to_list(data):
    new_data = []
    for item in data:
        
        new_data.append(ast.literal_eval(item))

    return new_data

def nested_list_to_np(data):
    new_data = np.array([])
    for item in data:
        for i in range(len(item)):
            
            if type(item[i]) == type([]):
                temp = triangle_data(item[i])
                item[i] = np.array(item[i], dtype=object)
            
        item = np.array(item, dtype=float)
    data = np.array(data, dtype=object)
    return data

def save_to_file(path, reference_video, lines=[]):
    
    strings = ['\n<ReferenceVideo: ' + str(reference_video) + '>'] + [str(x) for x in lines]
    if strings in read_file(path):
        print("Cannot write. The specified ReferenceVideo: '{}' has identical data in file '{}'.".format(reference_video, path))
    elif strings[0] in read_file(path):
        print("Cannot write. The specified ReferenceVideo: '{}' already exists and has different data in the file '{}'. Either delete this entry in the file or ignore your entry.".format(reference_video, path))
    else:
        with open(path, 'a') as f:
            f.writelines('\n'.join(strings))
            f.close()
        return True

def triangle_data(points):
    start = np.array(points[0])
    middle = np.array(points[ len(points)//2 ])
    end = np.array(points[-1])

    
    # Measure the euclidian traveled from start of life to end of life
    start_end_distance = dist(start, end)
    # Measure the difference between (start_end_distance) and (start_middle_end_distance)
    middle_point_deviation = gait(start, middle, end)
    # Measure the difference between (middle_end) and (start_middle)
    aprox_acceleration = accel(start, middle, end)

    return start_end_distance, middle_point_deviation, aprox_acceleration




def dist(start, end):
    distance = np.linalg.norm(start - end)
    return distance

# Determines how different a vertex is in length compared to hypotenuse
def gait(start, middle, end):
    # (|start-middle| + |middle-end|) - (|start-end|)
    deviation = ( dist(start, middle) + dist(middle, end) ) - dist(start, end)
    return deviation

# Determines if the car has increased or decreased in speed over time
def accel(start, middle, end):
    # (|middle - end| - |start - middle|)
    acceleration = dist(middle, end) - dist(start, middle)
    return acceleration


