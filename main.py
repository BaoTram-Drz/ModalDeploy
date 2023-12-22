import streamlit as st
import tensorflow as tf
import h5py
import numpy as np

# train
train_filename = 'model.hdf5'
datasets = []
with h5py.File(train_filename, "r") as f:
    for file_key in f.keys():
        group = f[file_key]
        if isinstance(group, h5py._hl.dataset.Dataset):
            datasets.append(np.array(group))
            continue
        for group_key in group.keys():
            group2 = group[group_key]
            if isinstance(group2, h5py._hl.dataset.Dataset):
                datasets.append(np.array(group2))
                continue
            for group_key2 in group2.keys():
                group3 = group2[group_key2]
                if isinstance(group3, h5py._hl.dataset.Dataset):
                    datasets.append(np.array(group3))
                    continue

print(len(datasets))
for dataset in datasets:
    print(dataset)