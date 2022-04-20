# Insights.py
# Produces insights into data that was extracted


import numpy as np
import read_write
import read_write
import os
import matplotlib.pyplot as plt
import pdb
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

main_path = os.path.dirname(os.path.abspath(__file__)) 
datapath = os.path.join(main_path, "Data", "Inflow")
train_labels_path = os.path.join(datapath, "training_labels.txt")
train_data_path = os.path.join(datapath, "training_data.txt")

def main():
    
    # Create synthetic data
    n = 40  # number of samples
    classes = 2  # number of classes
    seed = 6  # for repeatability
    x_raw, t_raw = get_training()
    
    # Remove -1's from the dataset
    t_raw = np.where(t_raw==1, t_raw, t_raw*0)

    x, xT, t, tT = train_test_split(x_raw, t_raw, test_size=0.33, random_state=42)
    
    
    # Train SVM
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(x, t)

    print("Training Accuracy: ", clf.score(x, t))
    print("Testing Accuracy:", clf.score(xT, tT))

    pdb.set_trace()


def get_training():
    
    data = np.array(read_write.read_file(train_data_path))
    labels = np.array(read_write.read_file(train_labels_path))
    
    data = read_write.purge_references(data)
    
    data = read_write.str_to_list(data)
    
    data = read_write.nested_list_to_np(data)


    labels = read_write.purge_references(labels)
    
    labels = read_write.str_to_list(labels)

    labels = np.array(labels)

    return data, labels


    




if __name__ == '__main__':
    main()