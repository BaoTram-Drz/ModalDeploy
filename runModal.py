import streamlit as st
import subprocess
import h5py

# Hàm kiểm tra và cài đặt thư viện h5py
def check_and_install_h5py():
    try:
        import h5py
    except ImportError:
        st.warning("Đang cài đặt thư viện h5py...")
        subprocess.call(['pip', 'install', 'h5py'])
        st.success("Cài đặt hoàn tất. Vui lòng chạy lại ứng dụng.")
        st.stop()

# Kiểm tra và cài đặt thư viện h5py
check_and_install_h5py()

# Hàm đọc file HDF5
def read_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            # Đọc dữ liệu từ file HDF5 ở đây
            # Ví dụ: Đọc dataset có tên là 'data'
            data = file['data'][:]
            return data
    except Exception as e:
        st.error(f"Lỗi khi đọc file HDF5: {str(e)}")
        return None

# Đường dẫn đến file HDF5 của bạn (ngang hàng với file script Python)
file_path = "modal.hdf5"

# Đọc dữ liệu từ file HDF5
modal_data = read_hdf5_file(file_path)

# Hiển thị dữ liệu bằng Streamlit
if modal_data is not None:
    st.write("Dữ liệu từ file modal.hdf5:")
    st.write(modal_data)
