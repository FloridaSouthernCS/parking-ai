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
import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn import svm
from skimage.feature import hog
from skimage.util import montage
from sklearn.metrics import plot_confusion_matrix

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

    mislabeled_train = np.where(clf.predict(x) != t)
    mislabeled_test = np.where(clf.predict(xT) != tT)
    
    print("\nCreating confusion matrix... ")
    plt.rc('font', size=6)
    plt.rc('figure', titlesize=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2, top=0.9, right=0.9, left=0.1)

    ax.set_title("SVM Confusion Matrix")
    cm = plot_confusion_matrix(clf, xT, tT,
                                normalize='all',
                                display_labels=[0, 1],
                                xticks_rotation='vertical',
                                cmap=plt.cm.Blues,
                                ax=ax)
    plt.show()

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