# Python program to demonstrate
# HDF5 file

import numpy as np
import h5py

# initializing a random numpy array
arr = np.random.randn(1000)

# creating a file
# with h5py.File('model.hdf5', 'w') as f: 
# 	dset = f.create_dataset("default", data = arr)
	
# open the file as 'f'
with h5py.File('model.hdf5', 'r') as f: 
	data = f['default']
	
	# get the minimum value
	print(min(data)) 
	
	# get the maximum value
	print(max(data))
	
	# get the values ranging from index 0 to 15
	print(data[:16])
	
