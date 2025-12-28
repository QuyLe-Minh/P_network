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

def plot_histogram(masked_inp, organ):
    # Calculate the histogram
    counts, bin_edges = np.histogram(masked_inp, bins=60)
    
    # Find the bin with the highest frequency
    max_count_index = np.argmax(counts)
    max_count = counts[max_count_index]
    bin_range = (bin_edges[max_count_index], bin_edges[max_count_index + 1])
    
    # Print the bin range and the count
    print(f'The bin with the highest frequency is {bin_range} with a count of {max_count}')
    
    # Plot the histogram
    plt.hist(masked_inp, bins=60, color='blue', edgecolor='black')
    plt.title(f'Histogram of {idx_to_organ[organ]} Values of Patient 8')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(-4, 4)
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
    
    
def extract_organs(organ, data_path, label_path, posenc=None):
    gt = load_nib(label_path).astype(np.uint8)
    inp = load_nib(data_path)
    inp = np.clip(inp, -958, 327)
    inp = (inp - 82.92) / 136.97    
    
    mask = (gt == organ).astype(np.uint8)
    masked_inp = inp[mask==1]
    
    print(f"Mean: {masked_inp.mean()}")
    print(f"Median: {np.median(masked_inp)}")
    print(f"Std: {masked_inp.std()}")  
    
    plot_histogram(masked_inp, organ)
    inp = (255 * (inp - inp.min()) / (inp.max() - inp.min())).astype(np.uint8)
    inp[mask!=1] = 0
    # display_organ(inp)
        
    
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
    
    organ = input("Enter organ: ")
    organ_id = organ_d[organ]
    
    extract_organs(organ_id, inp_path, gt_path)
    
    
