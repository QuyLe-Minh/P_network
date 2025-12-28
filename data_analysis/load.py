import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_nii_gz(file_path):
    # Load the .nii.gz file
    nii = nib.load(file_path)
    
    # Extract the data
    data = nii.get_fdata()
    
    # Print the maximum and minimum values
    print(f"Max value: {np.max(data)}")
    print(f"Min value: {np.min(data)}")
    plt.imshow(data[..., 15], cmap='gray')
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "C:/Users/Admin/Downloads/img0038.nii.gz"
    load_nii_gz(file_path)