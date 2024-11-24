# data_processing.py

import numpy as np
from skimage import measure

def correct_outside_skull_mask(outside_skull_mask):
    """
    Corrects the outside_skull_mask by removing any connected regions
    that are not connected to the boundary of the 3D volume.

    Parameters:
    ----------
    outside_skull_mask : numpy.ndarray
        A 3D binary numpy array where voxels with value 1 are flagged as outside the skull.

    Returns:
    -------
    corrected_mask : numpy.ndarray
        A 3D binary numpy array with the corrected outside skull mask.
    """
    # Ensure the input is a binary mask
    if not np.array_equal(outside_skull_mask, outside_skull_mask.astype(bool)):
        raise ValueError("Input mask must be binary (contain only 0s and 1s).")
    
    # Label connected components in the mask
    # connectivity=3 for 26-connectivity in 3D
    labeled_mask = measure.label(outside_skull_mask, connectivity=3)
    
    # Extract labels from all six faces of the 3D volume
    boundary_labels = np.unique(np.concatenate([
        labeled_mask[0, :, :],      # Front face
        labeled_mask[-1, :, :],     # Back face
        labeled_mask[:, 0, :],      # Left face
        labeled_mask[:, -1, :],     # Right face
        labeled_mask[:, :, 0],      # Top face
        labeled_mask[:, :, -1]      # Bottom face
    ]))
    
    # Remove the background label (assumed to be 0)
    boundary_labels = boundary_labels[boundary_labels != 0]
    
    # Create a mask that only includes labels connected to the boundary
    exterior_mask = np.isin(labeled_mask, boundary_labels)
    
    # Combine the original mask with the exterior mask to remove enclosed regions
    corrected_mask = np.logical_and(outside_skull_mask, exterior_mask).astype(np.int32)
    
    return corrected_mask
