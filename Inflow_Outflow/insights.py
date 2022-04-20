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

main_path = os.path.dirname(os.path.abspath(__file__)) 
datapath = os.path.join(main_path, "Data", "Inflow")
train_labels_path = os.path.join(datapath, "training_labels.txt")
train_data_path = os.path.join(datapath, "training_data.txt")

def main():
    
    # Create synthetic data
    n = 40  # number of samples
    classes = 2  # number of classes
    seed = 6  # for repeatability
    x, t = get_training()
    
    # Remove -1's from the dataset
    t = np.where(t==1, t, t*0)
    
    
    # Train SVM
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(x, t)

    # Show some stuff
    plt.scatter(x[:, 0], x[:, 1], c=t, s=30, cmap=plt.cm.Paired)

    # Compute decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X = np.linspace(xlim[0], xlim[1], 30)
    Y = np.linspace(ylim[0], ylim[1], 30)
    Xm, Ym = np.meshgrid(X, Y)
    Xtest = np.vstack([Xm.ravel(), Ym.ravel()]).T
    d = clf.decision_function(Xtest).reshape(Xm.shape)
    ax.contour(Xm, Ym, d,
        colors='k',
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=['--', '-', '--'])
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')
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