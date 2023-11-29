import os
import scipy.io
from scipy.interpolate import interp1d
import numpy as np
import h5py

def get_one_hot_enc(class_array):
    
    
    unique_classes = set(cls for record in class_array for cls in record)
    class_to_index = {cls: i for i, cls in enumerate(unique_classes)}

    # Step 2: Initialize one-hot encoded matrix
    num_records = len(class_array)
    num_classes = len(unique_classes)
    one_hot_encoded = np.zeros((num_records, num_classes), dtype=int)

    # Step 3: Populate the matrix
    for i, record in enumerate(class_array):
        for cls in record:
            one_hot_encoded[i, class_to_index[cls]] = 1
    return one_hot_encoded,class_to_index
    


def read_mat_file(file_path):
    # data = scipy.io.loadmat(file_path)
    # return next(iter(data.values()))
    data = scipy.io.loadmat(file_path)
    mat_data = data['val']
    
    # Convert to numpy array if not already
    if not isinstance(mat_data, np.ndarray):
        mat_data = np.array(mat_data,dtype='float16')

    # Interpolate if the second dimension is not 5000
    current_shape = mat_data.shape
#     print(mat_data)
    if current_shape[1] != 5000:
        # Initialize the array for interpolated data
        interpolated_data = np.zeros((12, 5000))

        # Interpolating each row
        x_original = np.linspace(0, 1, current_shape[1])
        x_new = np.linspace(0, 1, 5000)
        
        for i in range(12):
            f = interp1d(x_original, mat_data[i, :], kind='linear', fill_value="extrapolate")
            interpolated_data[i, :] = f(x_new)

        mat_data = interpolated_data

    return mat_data


def read_hea_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#Dx:"):
                return line.split(":")[1].strip().split(",")
    print("Returning None")
    return None

def process_directory_dataset1(directory):
    mat_data_list = []
    class_list = []
    count=0
    for root, dirs, files in os.walk(directory):
        if(count%1000==0):
            print("Processing ",count)
        for file in files:
            if file.endswith(".mat"):
                count+=1
                mat_file_path = os.path.join(root, file)
                mat_data = read_mat_file(mat_file_path)
                

                hea_file_path = os.path.splitext(mat_file_path)[0] + '.hea'
                if os.path.exists(hea_file_path):
                    class_data = read_hea_file(hea_file_path)
                    
                    # for class_d in class_data:
                    #     if(class_d not in class_include):
                    #         class_data.remove(class_d)
                    if(len(class_data)!=0 and len(class_data)!=None ):
                        mat_data_list.append(mat_data)
                        class_list.append(class_data)

    return mat_data_list, class_list

# Replace 'path_to_your_folder' with the path to your main directory
import pandas as pd
df = pd.read_csv("data_dist.csv")
# class_include = df.nlargest(n=10,columns=['values'])['class'].tolist()
mat_data, class_data = process_directory_dataset1("/pscratch/sd/k/ktub1999/289G/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords")
# mat_data2, class_data2= process_directory_dataset1("/pscratch/sd/k/ktub1999/289G/Training_WFDB")
# Convert lists to numpy arrays
# mat_data.extend(mat_data2)
# class_data.extend(class_data2)
mat_data_array = np.array(mat_data)
print(mat_data_array.shape)
# print(class_data.shape)
class_array_one_hot,class_to_index = get_one_hot_enc(class_data)
# class_data_array = np.array(class_data)

# mat_data_array and class_data_array now hold your data
with h5py.File('output_datasets_Single.hdf5', 'w') as hdf:
    hdf.create_dataset('data', data=mat_data_array)
    hdf.create_dataset('class', data=class_array_one_hot)
#     hdf.create_dataset('class_index', data=class_to_index)
import json
with open("indexSingle.json", "w") as f:
    json.dump(class_to_index, f, indent=4)
print(class_to_index)
print("Data and metadata have been stored in 'output.hdf5'")