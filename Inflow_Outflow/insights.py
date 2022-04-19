# Insights.py
# Produces insights into data that was extracted


import numpy as np
from read_write import nested_list_to_np
import read_write
import os
import pdb

main_path = os.path.dirname(os.path.abspath(__file__)) 
datapath = os.path.join(main_path, "Data", "Inflow")
train_labels_path = os.path.join(datapath, "training_labels.txt")
train_data_path = os.path.join(datapath, "training_data.txt")

def main():
    
    # Get data and labels for training
    xtrain, ytrain = get_training()

    # Get data and labels for testing
    xtest = None
    ytest = None













def get_training():
    data = []
    labels = []

    data = np.array(read_write.read_file(train_data_path))
    labels = np.array(read_write.read_file(train_labels_path))
    
    data = read_write.purge_references(data)

    data = read_write.str_to_list(data)
    data = nested_list_to_np(data)
    
    for i in range(len(data)):
        
        temp = read_write.triangle_data(data[i,0])
        temp2 = np.append(data[i], temp) 
        pdb.set_trace()
        data[i] = temp2
        
    


    return data, labels


    




if __name__ == '__main__':
    main()