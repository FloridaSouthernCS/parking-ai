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
import sklearn.metrics as sk
import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn import svm
from skimage.feature import hog
from skimage.util import montage

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
    
    # Separate out 2 classes of our choice. Options: 1(car), 0(not_car), -1(noise)

    # make all noise and not_car 0
    t_raw = np.where(t_raw==1, t_raw, t_raw*0) 
    labels = ['noise/not_car', 'car']

    

    # Get only cars and noise
    # t_raw = t_raw[np.where(t_raw!=0)] 
    # x_raw = x_raw[np.where(t_raw!=0)]
    # labels = ['noise/not_car', 'car']

    # split into training and testing
    x, xT, t, tT = train_test_split(x_raw, t_raw, test_size=0.33, random_state=18, stratify=t_raw)
    
    
    # Train SVM
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(x, t)

    # pdb.set_trace()
    x_pred = clf.predict(x)
    xT_pred = clf.predict(xT)

    print("Training Accuracy: ", clf.score(x, t))
    print("Testing Accuracy:", clf.score(xT, tT))


    precision = sk.precision_score(tT, xT_pred, labels=labels, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    recall = sk.recall_score(tT, xT_pred, labels=labels)
    
    print("Precison: ", precision)
    print("Recall: ", recall)

    

    
    accuracy = sk.accuracy_score(tT, xT_pred)

    mislabeled_train = np.where(x_pred != t)
    mislabeled_test = np.where(xT_pred != tT)

    print("\nCreating confusion matrix... ")
    plt.rc('font', size=6)
    plt.rc('figure', titlesize=10)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # plt.subplots_adjust(bottom=0.2, top=0.9, right=0.9, left=0.1)

    # ax.set_title("SVM Confusion Matrix")
    # cm = sk.plot_confusion_matrix(clf, xT, tT,
    #                             normalize='all',
    #                             display_labels=labels,
    #                             xticks_rotation='vertical',
    #                             cmap=plt.cm.Blues,
    #                             ax=ax)

    cm = sk.confusion_matrix(tT, xT_pred)
    cmd = sk.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cmd.plot()
    cmd.ax_.set_title("SVM Confusion Matrix")
    
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