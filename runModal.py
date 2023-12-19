import streamlit as st
import h5py

def read_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            # Read data from the HDF5 file here
            data = file['data'][:]
            return data
    except Exception as e:
        st.error(f"Error reading HDF5 file: {str(e)}")
        return None

# Allow users to upload the HDF5 file
uploaded_file = st.file_uploader("Upload HDF5 file", type=["hdf5", "h5"])

# Display data if the file is uploaded
if uploaded_file:
    # Save the uploaded file to a temporary location
    with open("temp.hdf5", "wb") as f:
        f.write(uploaded_file.read())
    # Read data from the HDF5 file
    modal_data = read_hdf5_file("temp.hdf5")
    # Display the data
    if modal_data is not None:
        st.write("Data from HDF5 file:")
        st.write(modal_data)
