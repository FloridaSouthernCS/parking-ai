# pca_knn.py
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
        result = input("Trackable: " + str(trackable.get_id()) + " is a car(c), not_car(n), noise([enter]): ")
        
        # Aquire training label
        if result == "c":
            labels.append(1)
        elif result == "n":
            labels.append(0)
        else:
            labels.append(-1)

        # Aquire training data
        temp_data = [trackable.get_life_func()]
        temp_data.append(trackable.get_func_contour_size(function=np.mean))
        temp_data.append(trackable.get_func_contour_size(function=np.median))
        temp_data.append(trackable.get_func_contour_size(function=np.max))

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

def save_to_file(path, reference_video, lines=[]):
    
    strings = ['<ReferenceVideo: ' + str(reference_video) + '>'] + [str(x) for x in lines]
    if strings in read_file(path):
        print("Cannot write. The specified ReferenceVideo: '{}' has identical data in file '{}'.".format(reference_video, path))
    elif strings[0] in read_file(path):
        print("Cannot write. The specified ReferenceVideo: '{}' already exists and has different data in the file '{}'. Either delete this entry in the file or ignore your entry.".format(reference_video, path))
    else:
        with open(path, 'a') as f:
            f.writelines('\n'.join(strings))
            f.close()
        return True
