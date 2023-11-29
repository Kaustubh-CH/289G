import h5py
import numpy as np

def check_nan_in_hdf5(file_path):
    with h5py.File(file_path, 'r') as file:
        for dataset_name in file:
            data = file[dataset_name][:]
            if np.isnan(data).any():
                print(f"NaN found in dataset '{dataset_name}'", np.isnan(data).sum())
            else:
                print(f"No NaN found in dataset '{dataset_name}'",data.shape)

# Replace 'path_to_your_hdf5_file.hdf5' with the actual path to your HDF5 file
check_nan_in_hdf5('/pscratch/sd/k/ktub1999/289G/output2_datasets_Norm_withoutZeros.hdf5')
