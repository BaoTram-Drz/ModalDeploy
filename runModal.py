import h5py

def read_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            # Đọc dữ liệu từ file HDF5 ở đây
            # Ví dụ: Đọc dataset có tên là 'data'
            data = file['data'][:]
            
            # Thực hiện các thao tác khác với dữ liệu
            
            print("Đọc file HDF5 thành công.")
            return data
    except Exception as e:
        print(f"Lỗi khi đọc file HDF5: {str(e)}")
        return None

if __name__ == "__main__":
    file_path = "modal.hdf5"  # Đường dẫn đến file HDF5 của bạn
    read_hdf5_file(file_path)
