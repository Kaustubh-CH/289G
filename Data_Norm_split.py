import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as file:
        data = file['data'][:]
        classes = file['class'][:]
    return data, classes
def normalize_data2(data):
    is_non_zero = ~(data == 0).all(axis=2).any(axis=1)
    data = data[is_non_zero]
   
    parts = 10
    for i in range(parts):
        mean = data[i*data.shape[0]//parts:(i+1)*data.shape[0]//parts].mean(axis=( 2), keepdims=True)
        std = data[i*data.shape[0]//parts:(i+1)*data.shape[0]//parts].std(axis=( 2), keepdims=True)
        
        data[i*data.shape[0]//parts:(i+1)*data.shape[0]//parts] = (data[i*data.shape[0]//parts:(i+1)*data.shape[0]//parts] - mean) / std
    
    

    return data,is_non_zero
def normalize_data(data):
    # Assuming data shape is (N, 12, 5000)
    # Normalize across the second dimension (12)
    row_has_zeros = np.any(np.all(data == 0, axis=2), axis=1)

    # Filter out those elements
    data = data[~row_has_zeros]

    mean = data.mean(axis=( 2), keepdims=True)
    std = data.std(axis=( 2), keepdims=True)
    normalized_data = (data - mean) / std

    normalized_data = np.zeros_like(data, dtype=np.float16)
    for i in range(data.shape[1]):
        normalized_data[:, i, :] = data[:, i, :]
        mean_X = np.mean(data[:,i,:],axis = 1)
        std_X = np.std(data[:,i,:],axis=1)
        mean_X = mean_X.reshape(mean_X.shape[0],1)
        std_X = mean_X.reshape(std_X.shape[0],1)
        # normalized_data[:, i, :] = zscore(data[:, i, :], axis=1, ddof=1).astype(np.float16)
        if(np.isnan(std_X).any()):
            print(i,"Nan in this segment")
            breakasdf
        for index,s in enumerate(std_X):
            if s ==0:
                print("Nan::",normalized_data[index,:,:])
                print("issue",i)
                std_X[index]=1
                

        if(np.isnan( normalized_data[:, i, :]).any()):
            print(i,"Nan in this segment")
            breakasdf
        try:
            normalized_data[:, i, :]=(normalized_data[:, i, :] - mean_X)/std_X
        except RuntimeWarning:
            asdfa
    return normalized_data

def split_data(data, classes, test_size=0.2, val_size=0.1):
    # Split data into training and test+validation
    X_train, X_temp, y_train, y_temp = train_test_split(data, classes, test_size=test_size + val_size)
    
    # Split the test+validation into test and validation
    val_relative_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_relative_size)

    return X_train, X_test, X_val, y_train, y_test, y_val

def save_hdf5(file_path, X_train, X_test, X_val, y_train, y_test, y_val):
    with h5py.File(file_path, 'w') as file:
        file.create_dataset('X_train', data=X_train)
        file.create_dataset('X_test', data=X_test)
        file.create_dataset('X_val', data=X_val)
        file.create_dataset('y_train', data=y_train)
        file.create_dataset('y_test', data=y_test)
        file.create_dataset('y_val', data=y_val)

# Main workflow
input_file_path = 'output2_datasets_Class10.hdf5'
output_file_path = 'output2_datasets_Class10NormNew.hdf5'

data, classes = read_hdf5(input_file_path)
data,exlude_index = normalize_data2(data)
classes = classes[exlude_index]
X_train, X_test, X_val, y_train, y_test, y_val = split_data(data, classes)
save_hdf5(output_file_path, X_train, X_test, X_val, y_train, y_test, y_val)
