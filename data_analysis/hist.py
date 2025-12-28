import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib

idx_to_organ = {
    1: 'Spleen',
    2: 'Right Kidney',
    3: 'Left Kidney',
    4: 'Gallbladder',
    6: 'Liver',
    7: 'Stomach',
    8: 'Aorta',
    11: 'Pancreas',
}

def load_nib(path):
    nii = nib.load(path)
    return nii.get_fdata()

def plot_histogram(masked_inp1, masked_inp2):
    # Calculate the histograms
    counts1, bin_edges1 = np.histogram(masked_inp1, bins=60)
    counts2, bin_edges2 = np.histogram(masked_inp2, bins=60)
    
    # Plot the histograms
    plt.hist(masked_inp1, bins=60, color='blue', edgecolor='black', alpha=0.5, label='Liver')
    plt.hist(masked_inp2, bins=60, color='red', edgecolor='black', alpha=0.5, label='Stomach')
    
    # Add titles and labels
    plt.title('Histogram of Intensities between Liver values and Stomach values of Patient 8')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(-2, 2)
    plt.legend()
    plt.show()
    
def display_organ(inp):
    h, w, d = inp.shape
    
    # Create a figure and an axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # Display the first slice
    img = ax.imshow(inp[:, :, 0], cmap='gray', vmin=0, vmax=255)
    
    # Create a slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Slice', 0, d-1, valinit=0, valstep=1)
    
    # Update function for the slider
    def update(val):
        slice_index = int(slider.val)
        img.set_data(inp[:, :, slice_index])
        fig.canvas.draw_idle()
    
    # Attach the update function to the slider
    slider.on_changed(update)
    
    # Show the plot with the slider
    plt.show()
    
    
def extract_organs(data_path, label_path):
    gt = load_nib(label_path).astype(np.uint8)
    inp = load_nib(data_path)
    inp = np.clip(inp, -958, 327)
    inp = (inp - 82.92) / 136.97    
    
    mask = (gt == 6).astype(np.uint8)
    masked_inp1 = inp[mask==1]
    
    mask = (gt == 7).astype(np.uint8)
    masked_inp2 = inp[mask==1]

    plot_histogram(masked_inp1, masked_inp2)
        
    
if __name__ == "__main__":
    organ_d = {
        'Spl': 1,
        'RKid': 2,
        'LKid': 3,
        'Gal': 4,
        'Liv': 6,
        'Sto': 7,
        'Aor': 8,
        'Pan': 11
    }
    
    id = input("Enter patient id: ")
    patient = id.zfill(4)
    
    inp_path = f"C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/inp/img{patient}.nii.gz"
    gt_path = f"C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/gt/test/img{patient}.nii.gz"
    
    extract_organs(inp_path, gt_path)
    
    
