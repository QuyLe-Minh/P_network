import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
import time

def load_nib(path):
    nii = nib.load(path)
    return nii.get_fdata()

def create_mask(image, pred, gt, organ_id):
    """
    Create mask for pred based on gt:

    TP: green
    FP: red
    FN: yellow
    
    Args:
        image (numpy array): (H, W) as an inp
        gt (numpy array): (H, W) from 0 to 13
        pred (numpy array): (H, W) from 0 to 13
        organ_id (int): 0 to 13

    Returns:
        image_with_mask (numpy array)
    """
    blue = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    red = (255, 0, 0)
    
    image_with_mask = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    y_hat = (pred == organ_id).astype(np.uint8)
    y = (gt == organ_id).astype(np.uint8)

    intersection = y_hat * y
    image_with_mask[intersection == 1] = np.array(green, np.uint8)
    FP = y_hat - intersection
    image_with_mask[FP == 1] = np.array(red, np.uint8)
    FN = y - intersection
    image_with_mask[FN == 1] = np.array(yellow, np.uint8)

    return image_with_mask

def analysis(inp_path, pred_path, gt_path, organ_id):
    gt = load_nib(gt_path).astype(np.uint8)
    pred = load_nib(pred_path).astype(np.uint8)
    # pred = gt
    inp = load_nib(inp_path)
    inp = np.clip(inp, -958, 327)
    inp = (inp - 82.92)/136.97
    
    print(np.unique(pred[gt == 3]))
    print("Mx", inp.max())
    print("Mn", inp.min())
    inp = (inp - inp.min()) / (inp.max() - inp.min()) * 255
    inp = inp.astype(np.uint8)
    print("Shape:", inp.shape)
    
    _,_,d = inp.shape
    
    images_with_mask = []
    
    s = time.time()
    for i in range(d):
        y_hat = pred[:,:,i]
        y = gt[:,:,i]
        
        image = inp[:,:,i]
        image_with_mask = create_mask(image, y_hat, y, organ_id)
        images_with_mask.append((image_with_mask, image))
        
    print(f"Process need {time.time()-s} seconds")
        
    return images_with_mask

def display(inp, pred_path, gt_path, organ_id):
    images = analysis(inp, pred_path, gt_path, organ_id)
    d = len(images)
        
    # Create a figure and an axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Display the first image
    img_display1 = ax1.imshow(images[0][1], cmap='gray')
    img_display2 = ax2.imshow(images[0][0], cmap='gray')

    # Create a slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, d-1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        frame = int(slider.val)
        img_display1.set_data(images[frame][1])
        img_display2.set_data(images[frame][0])
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()

def plot_gt(path):
    try:
        image = load_nib(path).astype(np.uint8)
        image = (image[:, :, 100] == 1).astype(np.uint8)
        plt.imshow(image * 100, cmap='gray')
        plt.show()
    except:
        print(f"Error with {path}")
    
if '__main__' == __name__:

    organ_d = {
        "BG": 0,
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
    id = str(id)
    patient = id.zfill(4)
    inp_path = f"C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/inp/img{patient}.nii.gz"
    # pred_path = f"C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/KL/img{patient}.nii.gz"
    # gt_path = f"C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/gt/test/img{patient}.nii.gz"
    pred_path = "C:/Users/Admin/Downloads/img0001.nii.gz"
    gt_path = "C:/Users/Admin/Downloads/img0001.nii.gz"
    organ = input("Enter organ: ")
    organ_id = organ_d[organ]

    display(inp_path, pred_path, gt_path, organ_id)

    # for i in range(1, 40+1):
    #     id = str(i).zfill(4)
    #     inp_path = f"C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Results/gt/img{id}.nii.gz"
    #     plot_gt(inp_path)
        
        