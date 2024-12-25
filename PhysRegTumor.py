#!/usr/bin/env python3
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
from scipy.ndimage import zoom, center_of_mass
from scipy.interpolate import griddata

import nibabel as nib
from FK import Solver

import argparse
import numpy as np
import os
from functools import partial
import odil
from odil.runtime import tf
from data_processing import correct_outside_skull_mask


printlog = odil.util.printlog

# Data tensors
global wm_data, gm_data, csf_data, segm_data, pet_data
global stored_affine

# Material properties
global dtype, gamma, gamma_ch
dtype = np.float32
gamma_ch = -2
gamma = tf.constant(1.0, dtype=dtype) * (-1.) * gamma_ch

# Weights for different components of the model
global BC_w, pde_w, balance_w, neg_w, core_w, edema_w, outside_w, params_w, symmetry_w
BC_w = 850
pde_w = 60000
balance_w = 215
neg_w = 80
core_w = 17.5
edema_w = 13.5
outside_w = 13.5
params_w = 98
symmetry_w = 3.2

# Diffusion and reaction parameters
global D_ch, R_ch, rho_ch, matter_th
D_ch = 0.13
R_ch = 10
rho_ch = 0.06
matter_th = 0.1

# Initial and threshold settings
global c_init, th_down_s, th_up_s
th_down_s = 0.22
th_up_s = 0.62

# Control points and tissue segmentation weights
global CM_pos

global pet_w
pet_w = 1.0

# Regularization factors for spatial and temporal components
global TS, kxreg, ktreg
kxreg = 11
ktreg = 80

def gauss_sol3d_tf(x, y, z, dx, dy, dz, init_scale):
    # Experimentally chosen
    Dt = 5.0
    M = 250

    # Apply scaling to the coordinates
    x_scaled = x * dx / init_scale
    y_scaled = y * dy / init_scale
    z_scaled = z * dz / init_scale

    # Gaussian function calculation
    gauss = M / tf.pow(4 * tf.constant(np.pi, dtype=x.dtype) * Dt, 3/2) * tf.exp(- (tf.pow(x_scaled, 2) + tf.pow(y_scaled, 2) + tf.pow(z_scaled, 2)) / (4 * Dt))

    # Apply thresholds
    gauss = tf.where(gauss > 0.1, gauss, tf.zeros_like(gauss))
    gauss = tf.where(gauss > 1, tf.ones_like(gauss, dtype=dtype), gauss)

    return gauss

def unet_loss(unet_data, c_field, th_low):
    # Convert unet_data numpy array to a TensorFlow tensor
    unet_data_tensor = tf.convert_to_tensor(unet_data, dtype=tf.float32)
    
    # Ensure c_field is a float tensor (in case it isn't)
    c_field = tf.cast(c_field, tf.float32)
    
    # Create a mask where unet_data is 1
    mask = tf.equal(unet_data_tensor, 1.0)
    
    # Calculate the part of c_field that is under the threshold where unet_data is 1
    under_threshold = tf.less(c_field, th_low)
    relevant_under_threshold = tf.logical_and(mask, under_threshold)
    
    # Calculate losses where the condition is met
    # Loss is linearly scaled with c_field, reaching maximum when c_field is 0
    losses = tf.where(relevant_under_threshold, (th_low - c_field) / th_low, 0.0)
    
    # Calculate the maximum possible loss
    max_possible_loss = tf.where(mask, 1.0, 0.0)
    
    # Compute the total loss and normalize by the maximum possible loss
    total_loss = tf.reduce_sum(losses)
    max_loss = tf.reduce_sum(max_possible_loss)
    
    # Normalize the total loss to be between 0 and 1
    normalized_loss = total_loss / max_loss
    
    return normalized_loss


def transform_c(c, mod):
    global outside_skull_mask

    # Use the outside_skull_mask to set c to zero outside of the skull.
    # This is achieved by multiplying c with the inverted mask (where outside of the skull is zero).
    c_masked = c * (1 - outside_skull_mask)
    # Zero the tumor cells for the initial time step.
    if mod == np:
        c_masked = np.concatenate([c_masked[:0], np.zeros_like(c_masked)[:1], c_masked[1:]], axis=0)
    else:
        c_masked = tf.concat([c_masked[:0], tf.zeros_like(c_masked)[:1], c_masked[1:]], axis=0)
    
    return c_masked


def transform_txyz(tx, ty, tz, x, y, z, mod):
    global outside_skull_mask
    nth = 0 #first time slice
    #nth = tx.shape[0] - 1 #last time slice
    if mod == np:
        # Fix the trajectories for the initial time step.
        tx = np.concatenate([tx[:nth], x[nth:nth+1], tx[nth + 1:]], axis=0)
        ty = np.concatenate([ty[:nth], y[nth:nth+1], ty[nth + 1:]], axis=0)
        tz = np.concatenate([tz[:nth], z[nth:nth+1], tz[nth + 1:]], axis=0)

        # Fix the spatial boundary particles in all three dimensions.
        tx = np.concatenate([x[:, :1, :, :], tx[:, 1:-1, :, :], x[:, -1:, :, :]], axis=1)
        ty = np.concatenate([y[:, :, :1, :], ty[:, :, 1:-1, :], y[:, :, -1:, :]], axis=2)
        tz = np.concatenate([z[:, :, :, :1], tz[:, :, :, 1:-1], z[:, :, :, -1:]], axis=3)

        # Create tensors of the initial positions for particles outside the skull in all three dimensions.
        x_masked = x * outside_skull_mask
        y_masked = y * outside_skull_mask
        z_masked = z * outside_skull_mask

        # Use the mask to combine the fixed and moving portions in all three dimensions.
        tx_fixed = tx * (1-outside_skull_mask) + x_masked
        ty_fixed = ty * (1-outside_skull_mask) + y_masked
        tz_fixed = tz * (1-outside_skull_mask) + z_masked
    else:
        # Fix the trajectories for the initial time step.
        tx = tf.concat([tx[:nth], x[nth:nth+1], tx[nth + 1:]], axis=0)
        ty =  tf.concat([ty[:nth], y[nth:nth+1], ty[nth + 1:]], axis=0)
        tz =  tf.concat([tz[:nth], z[nth:nth+1], tz[nth + 1:]], axis=0)

        # Fix the spatial boundary particles in all three dimensions.
        tx =  tf.concat([x[:, :1, :, :], tx[:, 1:-1, :, :], x[:, -1:, :, :]], axis=1)
        ty =  tf.concat([y[:, :, :1, :], ty[:, :, 1:-1, :], y[:, :, -1:, :]], axis=2)
        tz =  tf.concat([z[:, :, :, :1], tz[:, :, :, 1:-1], z[:, :, :, -1:]], axis=3)

        # Create tensors of the initial positions for particles outside the skull in all three dimensions.
        x_masked = x * outside_skull_mask
        y_masked = y * outside_skull_mask
        z_masked = z * outside_skull_mask

        # Use the mask to combine the fixed and moving portions in all three dimensions.
        tx_fixed = tx * (1-outside_skull_mask) + x_masked
        ty_fixed = ty * (1-outside_skull_mask) + y_masked
        tz_fixed = tz * (1-outside_skull_mask) + z_masked

    return tx_fixed, ty_fixed, tz_fixed

def transform_txyz2(tx, ty, tz, x, y, z, mod):
    global outside_skull_mask
    #nth = 0 #first time slice
    nth = tx.shape[0] - 1 #last time slice
    if mod == np:
        # Fix the trajectories for the initial time step.
        tx = np.concatenate([tx[:nth], x[nth:nth+1], tx[nth + 1:]], axis=0)
        ty = np.concatenate([ty[:nth], y[nth:nth+1], ty[nth + 1:]], axis=0)
        tz = np.concatenate([tz[:nth], z[nth:nth+1], tz[nth + 1:]], axis=0)


    else:
        # Fix the trajectories for the initial time step.
        tx = tf.concat([tx[:nth], x[nth:nth+1], tx[nth + 1:]], axis=0)
        ty =  tf.concat([ty[:nth], y[nth:nth+1], ty[nth + 1:]], axis=0)
        tz =  tf.concat([tz[:nth], z[nth:nth+1], tz[nth + 1:]], axis=0)


    return tx, ty, tz


def transform_txyz2(tx, ty, tz, x, y, z, mod):
    return x, y, z


def compute_displacement_np(tx, ty, tz):
    """
    Compute the displacement vector fields from particle trajectories in 3D using NumPy.

    Parameters:
    tx, ty, tz: np.ndarray
        Particle trajectories, each with shape (nt, nx, ny, nz).

    Returns:
    ux, uy, uz: np.ndarray
        Displacement fields in x, y, and z direction, each with shape (nt, nx, ny, nz).
    """
    # Compute the initial positions in x, y, and z dimensions
    initial_position_x = tx[0, :, :, :]
    initial_position_y = ty[0, :, :, :]
    initial_position_z = tz[0, :, :, :]
    
    # Use broadcasting to compute displacements for all time steps in x, y, and z dimensions
    ux = tx - initial_position_x[np.newaxis, ...]
    uy = ty - initial_position_y[np.newaxis, ...]
    uz = tz - initial_position_z[np.newaxis, ...]

    return ux, uy, uz


def gradient_np(array, step, axis):
    """
    Compute the gradient of a NumPy array using a central difference scheme.
    """
    shifted_forward = np.roll(array, -1, axis=axis)
    shifted_backward = np.roll(array, 1, axis=axis)
    gradient = (shifted_forward - shifted_backward) / (2 * step)
    return gradient

def compute_strain_tensor_lagrangian_full_np(u_x, u_y, u_z, dx, dy, dz):
    """
    Compute the full 3D Green-Lagrange strain tensor E using NumPy arrays.

    Parameters
    ----------
    u_x, u_y, u_z : np.ndarray
        The components of the displacement field. Each should have shape (nt, nx, ny, nz).
    dx, dy, dz : float
        The step sizes in the x, y, and z directions.

    Returns
    -------
    np.ndarray
        The Green-Lagrange strain tensor. Shape: (3, 3, nt, nx, ny, nz).
    """
    # Compute the spatial gradients of the displacement fields
    u_x_x = gradient_np(u_x, dx, axis=1)
    u_x_y = gradient_np(u_x, dy, axis=2)
    u_x_z = gradient_np(u_x, dz, axis=3)
    u_y_x = gradient_np(u_y, dx, axis=1)
    u_y_y = gradient_np(u_y, dy, axis=2)
    u_y_z = gradient_np(u_y, dz, axis=3)
    u_z_x = gradient_np(u_z, dx, axis=1)
    u_z_y = gradient_np(u_z, dy, axis=2)
    u_z_z = gradient_np(u_z, dz, axis=3)


    # Compute the Green-Lagrange strain tensor components
    E_xx = 0.5 * (u_x_x + u_x_x + u_x_x * u_x_x + u_y_x * u_y_x + u_z_x * u_z_x)
    E_yy = 0.5 * (u_y_y + u_y_y + u_x_y * u_x_y + u_y_y * u_y_y + u_z_y * u_z_y)
    E_zz = 0.5 * (u_z_z + u_z_z + u_x_z * u_x_z + u_y_z * u_y_z + u_z_z * u_z_z)
    E_xy = 0.5 * (u_x_y + u_y_x + u_x_x * u_x_y + u_y_x * u_y_y)
    E_xz = 0.5 * (u_x_z + u_z_x + u_x_x * u_x_z + u_z_x * u_z_z)
    E_yz = 0.5 * (u_y_z + u_z_y + u_y_x * u_y_z + u_z_y * u_z_z)

    # Combine strain tensor components into a single 6D array
    E = np.array([[E_xx, E_xy, E_xz], [E_xy, E_yy, E_yz], [E_xz, E_yz, E_zz]])
    return E



def compute_displacement(tx, ty, tz):
    """
    Compute the displacement vector fields from particle trajectories in 3D.

    Parameters:
    tx, ty, tz: tf.Tensor
        Particle trajectories, each with shape (nt, nx, ny, nz).

    Returns:
    ux, uy, uz: tf.Tensor
        Displacement fields in x, y, and z direction, each with shape (nt, nx, ny, nz).
    """
    # Compute the initial positions in x, y, and z dimensions
    initial_position_x = tx[0, :, :, :]
    initial_position_y = ty[0, :, :, :]
    initial_position_z = tz[0, :, :, :]
    
    # Expand dimensions of initial positions to match shape of tx, ty, tz
    initial_position_x = tf.expand_dims(initial_position_x, axis=0)
    initial_position_y = tf.expand_dims(initial_position_y, axis=0)
    initial_position_z = tf.expand_dims(initial_position_z, axis=0)
    
    # Use broadcasting to compute displacements for all time steps in x, y, and z dimensions
    ux = tx - initial_position_x
    uy = ty - initial_position_y
    uz = tz - initial_position_z

    return ux, uy, uz

def gradient(tensor, step, axis, final_op=False):
    """
    Compute the gradient of a tensor using a central difference scheme in the interior and a first-order 
    scheme at the boundaries for 3D data.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to differentiate. Shape: (nt, nx, ny, nz).
    step : float
        The step size.
    axis : int
        The axis along which to compute the gradient.
    final_op : bool
        If this gradient operation is the final operation. Default is False.

    Returns
    -------
    tf.Tensor
        The gradient of the input tensor.
    """
    tensor_before = tf.roll(tensor, shift=1, axis=axis)
    tensor_after = tf.roll(tensor, shift=-1, axis=axis)

    # Central difference in the interior of the domain
    gradient = (tensor_after - tensor_before) / (2 * step)

    if axis == 1:
        # Forward difference at the left boundary and backward difference at the right boundary
        gradient_left = (tensor[:, 1, :, :] - tensor[:, 0, :, :]) / step
        gradient_right = (tensor[:, -1, :, :] - tensor[:, -2, :, :]) / step
        gradient = tf.concat([gradient_left[:, None, :, :], gradient[:, 1:-1, :, :], gradient_right[:, None, :, :]], axis=axis)
    
    elif axis == 2:
        # Forward difference at the top boundary and backward difference at the bottom boundary
        gradient_top = (tensor[:, :, 1, :] - tensor[:, :, 0, :]) / step
        gradient_bottom = (tensor[:, :, -1, :] - tensor[:, :, -2, :]) / step
        gradient = tf.concat([gradient_top[:, :, None, :], gradient[:, :, 1:-1, :], gradient_bottom[:, :, None, :]], axis=axis)

    elif axis == 3:
        # Forward difference at the front boundary and backward difference at the back boundary
        gradient_front = (tensor[:, :, :, 1] - tensor[:, :, :, 0]) / step
        gradient_back = (tensor[:, :, :, -1] - tensor[:, :, :, -2]) / step
        gradient = tf.concat([gradient_front[:, :, :, None], gradient[:, :, :, 1:-1], gradient_back[:, :, :, None]], axis=axis)

    return gradient


def dice_score_tf(mask1, mask2):
    # Convert boolean masks to a dtype that TensorFlow operations can work with
    mask1_float = tf.cast(mask1, dtype)
    mask2_float = tf.cast(mask2, dtype)

    # Compute the intersection
    intersection = tf.reduce_sum(mask1_float * mask2_float)

    # Compute Dice score
    return 2. * intersection / (tf.reduce_sum(mask1_float) + tf.reduce_sum(mask2_float))

def calculate_dice_scores(segm, coeff, c_euler_slice):
    # Create masks from c_euler_slice
    edema_mask_pred = (c_euler_slice > coeff[5]) & (c_euler_slice <= coeff[6])
    core_mask_pred = c_euler_slice >= coeff[6]

    # Convert these masks to TensorFlow tensors if necessary
    edema_mask_pred_tf = tf.convert_to_tensor(edema_mask_pred, dtype=tf.bool)
    core_mask_pred_tf = tf.convert_to_tensor(core_mask_pred, dtype=tf.bool)

    # Calculate masks from your segmentation
    edema_mask_true = get_edema_mask(segm)
    core_mask_true = get_core_mask(segm)

    # Calculate Dice scores
    dice_score_edema = dice_score_tf(edema_mask_true, edema_mask_pred_tf)
    dice_score_core = dice_score_tf(core_mask_true, core_mask_pred_tf)

    return dice_score_edema, dice_score_core

def get_core_mask(segm,mod=tf):
    # Use TensorFlow operations for logical or and equality checks
    return tf.logical_or(tf.equal(segm, 1), tf.equal(segm, 4))

def get_edema_mask(segm,mod=tf):
    # Use TensorFlow operation for equality check
    return tf.equal(segm, 3)


def get_core_loss_tf(c, th_up, segm):
    core_mask = get_core_mask(segm)

    # Compute the core loss where core_mask is True, else set to 0
    core_loss = tf.where(core_mask,
                         tf.clip_by_value(th_up - c, clip_value_min=0, clip_value_max=tf.float32.max),
                         tf.zeros_like(core_mask, dtype=tf.float32))

    return core_loss

def get_edema_loss_tf(c, th_down, th_up, segm):
    edema_mask = get_edema_mask(segm)

    # Define the condition for values within the desired range
    within_range_condition = tf.logical_and(c >= th_down, c <= th_up)

    # Combine the mask with the condition
    final_mask = tf.logical_and(edema_mask, tf.logical_not(within_range_condition))

    # Compute the edema loss
    edema_loss = tf.where(final_mask,
                          tf.abs(c - th_down) + tf.abs(c - th_up),
                          tf.zeros_like(final_mask, dtype=tf.float32))

    return edema_loss

def get_outside_segm_mask(segm, mod=tf):
    # Use TensorFlow operation for equality check
    return tf.equal(segm, 0)

def get_outside_segm_loss_tf(c, th_down, segm):
    outside_segm_mask = get_outside_segm_mask(segm)

    # Define the condition where c[-1] is not below th_down
    not_below_condition = c >= th_down

    # Combine the mask with the condition
    final_mask = tf.logical_and(outside_segm_mask, not_below_condition)

    # Compute the outside segment loss
    # If c[-1] is not below th_down in the outside_segm, penalize by the difference
    outside_segm_loss = tf.where(final_mask,
                                 c - th_down,
                                 tf.zeros_like(final_mask, dtype=tf.float32))

    return outside_segm_loss

def pet_loss(pet_data, segm_data, c_euler):
    # Create a mask where segm_data is 1 or 3
    mask = tf.logical_or(segm_data == 1, segm_data == 3)
    
    # Apply the mask to flatten only the selected voxels
    pet_data_masked = tf.boolean_mask(pet_data, mask)
    c_euler_masked = tf.boolean_mask(c_euler, mask)

    # Ensure both tensors are of the same data type
    pet_data_masked = tf.cast(pet_data_masked, dtype=tf.float32)
    c_euler_masked = tf.cast(c_euler_masked, dtype=tf.float32)
    
    # Compute Pearson correlation on the selected voxels
    def pearson_correlation(x, y):
        mean_x = tf.reduce_mean(x)
        mean_y = tf.reduce_mean(y)
        normalized_x = x - mean_x
        normalized_y = y - mean_y
        covariance = tf.reduce_sum(normalized_x * normalized_y)
        std_dev_x = tf.sqrt(tf.reduce_sum(tf.square(normalized_x)))
        std_dev_y = tf.sqrt(tf.reduce_sum(tf.square(normalized_y)))
        return covariance / (std_dev_x * std_dev_y)
    
    if tf.size(pet_data_masked) == 0 or tf.size(c_euler_masked) == 0:
        return 1.0  # If no valid voxels, return maximum loss
    correlation = pearson_correlation(pet_data_masked, c_euler_masked)
    return 1 - correlation  # loss is 1 minus the correlation coefficient



def compute_strain_tensor_lagrangian_full(u_x, u_y, u_z, domain):
    """
    Compute the full 3D Green-Lagrange strain tensor E given the x, y, and z components of the displacement field.

    Parameters
    ----------
    u_x : tf.Tensor
        The x-component of the displacement field. Shape: (nt, nx, ny, nz).
    u_y : tf.Tensor
        The y-component of the displacement field. Shape: (nt, nx, ny, nz).
    u_z : tf.Tensor
        The z-component of the displacement field. Shape: (nt, nx, ny, nz).

    Returns
    -------
    E : tf.Tensor
        The Green-Lagrange strain tensor. Shape: (3, 3, nt, nx, ny, nz).
    """
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')

    # Compute the spatial gradients of the displacement fields
    u_x_x = gradient(u_x, dx, axis=1)  # partial derivative of u_x w.r.t. x
    u_x_y = gradient(u_x, dy, axis=2)  # partial derivative of u_x w.r.t. y
    u_x_z = gradient(u_x, dz, axis=3)  # partial derivative of u_x w.r.t. z
    u_y_x = gradient(u_y, dx, axis=1)  # partial derivative of u_y w.r.t. x
    u_y_y = gradient(u_y, dy, axis=2)  # partial derivative of u_y w.r.t. y
    u_y_z = gradient(u_y, dz, axis=3)  # partial derivative of u_y w.r.t. z
    u_z_x = gradient(u_z, dx, axis=1)  # partial derivative of u_z w.r.t. x
    u_z_y = gradient(u_z, dy, axis=2)  # partial derivative of u_z w.r.t. y
    u_z_z = gradient(u_z, dz, axis=3)  # partial derivative of u_z w.r.t. z

    # Compute the Green-Lagrange strain tensor components
    E_xx = 0.5 * (u_x_x + u_x_x + u_x_x*u_x_x + u_y_x*u_y_x + u_z_x*u_z_x)
    E_yy = 0.5 * (u_y_y + u_y_y + u_x_y*u_x_y + u_y_y*u_y_y + u_z_y*u_z_y)
    E_zz = 0.5 * (u_z_z + u_z_z + u_x_z*u_x_z + u_y_z*u_y_z + u_z_z*u_z_z)
    E_xy = 0.5 * (u_x_y + u_y_x + u_x_x*u_x_y + u_y_x*u_y_y)
    E_xz = 0.5 * (u_x_z + u_z_x + u_x_x*u_x_z + u_z_x*u_z_z)
    E_yz = 0.5 * (u_y_z + u_z_y + u_y_x*u_y_z + u_z_y*u_z_z)

    # Combine strain tensor components into a single 6D array
    E = tf.stack([[E_xx, E_xy, E_xz], [E_xy, E_yy, E_yz], [E_xz, E_yz, E_zz]])

    return E


def get_diffusion_coefficient(wm_intensity, gm_intensity, D_s, R):
    return D_s * wm_intensity + (D_s / R) * gm_intensity

def m_Tildas(WM,GM,th):
        
    WM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(WM,-1,axis=0) + WM)/2,0)
    WM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(WM,-1,axis=1) + WM)/2,0)
    WM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(WM,-1,axis=2) + WM)/2,0)

    GM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(GM,-1,axis=0) + GM)/2,0)
    GM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(GM,-1,axis=1) + GM)/2,0)
    GM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(GM,-1,axis=2) + GM)/2,0)
    
    return {"WM_t_x": WM_tilda_x,"WM_t_y": WM_tilda_y,"WM_t_z": WM_tilda_z,"GM_t_x": GM_tilda_x,"GM_t_y": GM_tilda_y,"GM_t_z": GM_tilda_z}

def get_D(WM, GM, th, Dw, Dw_ratio):
    M = m_Tildas(WM,GM,th)
    D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
    D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    D_minus_z = Dw*(M["WM_t_z"] + M["GM_t_z"]/Dw_ratio)
    
    D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
    D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
    D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
    
    return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}


def operator_adv(ctx):
    global gamma, BC_w, pde_w, balance_w, neg_w, D_ch, R_ch, outside_skull_mask, neg_w,CM_pos, pet_w
    dt = ctx.step('t')
    dx = ctx.step('x')
    dy = ctx.step('y')
    dz = ctx.step('z')
    x = ctx.points('x')
    y = ctx.points('y')
    z = ctx.points('z')

    nt = ctx.size('t')
    nx = ctx.size('x')
    ny = ctx.size('y')
    nz = ctx.size('z')
    

    def single_var(key, st=0, sx=0, sy=0, sz=0):
        u = ctx.field(key, st, sx, sy, sz)
        return u

    
    def field_to_particles_3d(q_src, it):
        q_src = ctx.cast(q_src)
        # Pad the field for 3 dimensions
        q_src = pad_linear(q_src, [(1, 1), (1, 1), (1, 1)])

        # Initialize the tensor for the particle field
        qp = tf.zeros(tix[it].shape, dtype=q_src.dtype)

        # Loop through all combinations of x, y, z coordinates and weights
        for jx, jy, jz, jw in [
            (tix, tiy, tiz, sx0 * sy0 * sz0),
            (tixp, tiy, tiz, sx1 * sy0 * sz0),
            (tix, tiyp, tiz, sx0 * sy1 * sz0),
            (tixp, tiyp, tiz, sx1 * sy1 * sz0),
            (tix, tiy, tizp, sx0 * sy0 * sz1),
            (tixp, tiy, tizp, sx1 * sy0 * sz1),
            (tix, tiyp, tizp, sx0 * sy1 * sz1),
            (tixp, tiyp, tizp, sx1 * sy1 * sz1),
        ]:
            idx = tf.stack([jx[it] + 1, jy[it] + 1, jz[it] + 1], axis=-1)
            qp += jw[it] * tf.gather_nd(q_src, idx)
        return qp

    def laplace(st):
        """
        Calculate the Laplacian of a 4D field (time, x, y, z).
        
        :param st: A tuple of field values (q, qxm, qxp, qym, qyp, qzm, qzp)
        :return: Laplacian of the field
        """
        q, qxm, qxp, qym, qyp, qzm, qzp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        q_zz = (qzp - 2 * q + qzm) / dz**2
        q_lap = q_xx + q_yy + q_zz
        return q_lap

    def pad_linear(q, paddings):
        """
        Apply linear padding to a 4D field.
        
        :param q: The field to be padded
        :param paddings: Padding specifications
        :return: Padded field
        """
        qr = tf.pad(q, paddings, mode='reflect')
        qs = tf.pad(q, paddings, mode='symmetric')
        q_padded = 2 * qs - qr
        return q_padded

    def depad(q, paddings):
        """
        Remove padding from a 4D field.
        
        :param q: The padded field
        :param paddings: Padding specifications used for padding the field
        :return: Field with padding removed
        """
        pt, px, py, pz = paddings
        slices = tuple(slice(p[0], -p[1] if p[1] else None) for p in paddings)
        return q[slices]

    def laplace_roll(q):
        """
        Calculate the Laplacian of a 4D field using rolling operations.
        
        :param q: The field for which the Laplacian is to be calculated
        :return: Laplacian of the field
        """
        paddings = [[0, 0], [1, 1], [1, 1], [1, 1]]
        q_padded = pad_linear(q, paddings)
        qxm = tf.roll(q_padded, shift=1, axis=1)
        qxp = tf.roll(q_padded, shift=-1, axis=1)
        qym = tf.roll(q_padded, shift=1, axis=2)
        qyp = tf.roll(q_padded, shift=-1, axis=2)
        qzm = tf.roll(q_padded, shift=1, axis=3)
        qzp = tf.roll(q_padded, shift=-1, axis=3)
        laplacian = laplace((q_padded, qxm, qxp, qym, qyp, qzm, qzp))
        return depad(laplacian, paddings)


    # Unknown parameters
    coeff = ctx.field('coeff')

    res = []

    # Trajectories.
    tx = single_var('x')
    ty = single_var('y')
    tz = single_var('z')
    tx, ty, tz = transform_txyz(tx, ty, tz, x, y, z, tf)
    
    # Tumor
    c = single_var('c')
    c = transform_c(c, mod=tf)

    # Cell indices.
    dtx = tx / dx - 0.5
    dty = ty / dy - 0.5
    dtz = tz / dz - 0.5
    tix = tf.clip_by_value(ctx.cast(tf.floor(dtx), tf.int32), -1, nx - 1)
    tiy = tf.clip_by_value(ctx.cast(tf.floor(dty), tf.int32), -1, ny - 1)
    tiz = tf.clip_by_value(ctx.cast(tf.floor(dtz), tf.int32), -1, nz - 1)
    tixp = tix + 1
    tiyp = tiy + 1
    tizp = tiz + 1

    # Weights.
    sx1 = tf.clip_by_value(dtx - ctx.cast(tix), 0, 1)
    sy1 = tf.clip_by_value(dty - ctx.cast(tiy), 0, 1)
    sz1 = tf.clip_by_value(dtz - ctx.cast(tiz), 0, 1)  
    sx0 = 1 - sx1
    sy0 = 1 - sy1
    sz0 = 1 - sz1
   
    # Get white matter intensities at particle locations


    D_scalar = coeff[0]
    rho = coeff[1]
    x0 = coeff[2]
    y0 = coeff[3]
    z0 = coeff[4]
    gamma = tf.constant(1.0,dtype=dtype)*coeff[7]
    th_down = coeff[5]
    th_up = coeff[6]
    
    # Clip th_down values
    # Ensuring th_down is no less than 0.20 and no more than 0.35
    th_down = tf.clip_by_value(coeff[5], clip_value_min=0.20, clip_value_max=0.35)

    # Clip th_up values
    # Ensuring th_up is no less than 0.50 and no more than 0.85
    th_up = tf.clip_by_value(coeff[6], clip_value_min=0.50, clip_value_max=0.85)
    
    
    # Get the spatially-varying diffusion coefficient based on wm_intensities
    D = get_D(wm_data, gm_data, matter_th, D_scalar, R_ch)
    
    # Calculate the tumor pde loss
    pde_loss = tumor_pde_loss(tx, ty, tz, dt, c, D,rho)
    res += [pde_loss*pde_w]
    
    #BC
    c_init = gauss_sol3d_tf((x[0,:]-x0),(y[0,:]-y0),(z[0,:]-z0),dx,dy,dz,init_scale=0.8)
    #c_init_lagrange = field_to_particles_3d(c_init,1)
    c_init_lagrange = c_init #better for the init tissues
    bc = c[1] - c_init_lagrange
    res += [bc*BC_w]
    bc0 = c[0]
    res += [bc0*BC_w]
    

    
    # Calculate the strain tensors and balace them
    ux, uy, uz = compute_displacement(tx, ty, tz)
    E = compute_strain_tensor_lagrangian_full(ux, uy, uz, ctx)
    
    
    lambda_vals = compute_lambda(wm_data, gm_data, csf_data, c)
    lambda_vals = lambda_vals / tf.reduce_max(lambda_vals)
    mu_vals = compute_mu(wm_data, gm_data, csf_data, c)
    mu_vals = mu_vals / tf.reduce_max(mu_vals)
    
    #calculate the balance residual
    res_balance = compute_strain_balance_tf(E,c,gamma,lambda_vals,mu_vals,ctx)
    res += [res_balance*balance_w]
    
    #static tissues:
    #res += [[tx - x,ty - y,tz - z]]*w_static_tissues
    #negative tumor
    res +=  [(c - tf.abs(c))*neg_w]
    
    #outside of matter mask
    combined_matter = wm_data + gm_data
    combined_matter_mask = (combined_matter < matter_th).astype(int)
    # Repeat the combined_matter_mask across the time dimension
    combined_matter_mask_4d = np.repeat(combined_matter_mask[np.newaxis, :, :], nt, axis=0)
    res += [c * combined_matter_mask_4d * outside_w]
    
    #Data fit
    c_euler = particles_to_field_3d_average(c[-1],tx[-1],ty[-1],tz[-1],ctx.domain)
    res += [get_outside_segm_loss_tf(c_euler,th_down,segm_data)*outside_w]
    res += [get_edema_loss_tf(c_euler,th_down,th_up,segm_data)*edema_w]
    res += [get_core_loss_tf(c_euler,th_up,segm_data)*core_w]
    

    
    res += [tf.clip_by_value(0.11 - D_scalar ,0,100)*params_w]
    res += [tf.clip_by_value(0.02 - rho ,0,100)*params_w]
    res += [tf.clip_by_value(1.5 + gamma ,0,100)*params_w]
    
    # Smoothness of particles in space.
    ltx = laplace_roll(tx) * kxreg
    lty = laplace_roll(ty) * kxreg
    ltz = laplace_roll(tz) * kxreg
    res += [ltx, lty, ltz]

    # Smoothness of particles in time.
    txm = tf.roll(tx, shift=[1], axis=[0])
    tym = tf.roll(ty, shift=[1], axis=[0])
    tzm = tf.roll(tz, shift=[1], axis=[0])  

    txp = tf.roll(tx, shift=[-1], axis=[0])
    typ = tf.roll(ty, shift=[-1], axis=[0])
    tzp = tf.roll(tz, shift=[-1], axis=[0])  

    # Calculate residuals for x, y, and z dimensions
    res += [
        ((txp - 2 * tx + txm) / dt**2)[1:-1] * ktreg,
        ((typ - 2 * ty + tym) / dt**2)[1:-1] * ktreg,
        ((tzp - 2 * tz + tzm) / dt**2)[1:-1] * ktreg,  
    ]
    
    #brain looks simmetric at the beginning
    healthy_wm = field_to_particles_3d(wm_data,nt-1)
    for factor in [4, 8]:
        res += [calculate_symmetry_loss(healthy_wm, scale_factor=factor)*symmetry_w]

    # Process gray matter (gm)
    healthy_gm = field_to_particles_3d(gm_data,nt-1)
    for factor in [4, 8]:
        res += [calculate_symmetry_loss(healthy_gm, scale_factor=factor)*symmetry_w]

    # Process cerebrospinal fluid (csf)
    healthy_csf = field_to_particles_3d(csf_data,nt-1)
    for factor in [4, 8]:
        res += [calculate_symmetry_loss(healthy_csf, scale_factor=factor)*symmetry_w]

    # Parameter loss
    

    
    # normalize the loss constructed so far 
    kappa = 3.0 
    res = [x / kappa  for x in res]
    
    # PET loss
    res += [pet_loss(pet_data, segm_data, c_euler)*pet_w]
    
    return res

def calculate_symmetry_loss(healthy, scale_factor=1):
    # Assuming healthy shape is [depth, height, width]
    depth, height, width = healthy.shape

    # Downsample the tensor if scale_factor is greater than 1
    if scale_factor > 1:
        # Define the pooling size and strides for 3D data (batch, depth, height, width, channels)
        pool_size = [1, 1, scale_factor, scale_factor, 1]  # [batch, depth, height, width, channels]
        strides = [1, 1, scale_factor, scale_factor, 1]

        # Reshape the tensor to 5D for pooling (batch_size, depth, height, width, channels)
        healthy_reshaped = tf.reshape(healthy, [1, depth, height, width, 1])
        
        # Perform average pooling
        healthy_downsampled = tf.nn.avg_pool3d(input=healthy_reshaped, ksize=pool_size, strides=strides, padding='VALID')
        new_depth, new_height, new_width = healthy_downsampled.shape[1:4]
        healthy_downsampled = tf.reshape(healthy_downsampled, [new_depth, new_height, new_width])
    else:
        healthy_downsampled = healthy
        new_depth, new_height, new_width = healthy.shape

    # Update dimensions after downsampling
    _, height_downsampled, _ = healthy_downsampled.shape
    
    # Split the tensor into upper and lower halves along the Y-axis (height)
    mid = height_downsampled // 2
    upper_half = healthy_downsampled[:, :mid, :]
    lower_half = healthy_downsampled[:, mid:height_downsampled, :] if height_downsampled % 2 == 0 else healthy_downsampled[:, mid+1:, :]
    
    # Mirror the lower half for comparison
    lower_half_mirrored = tf.reverse(lower_half, axis=[1])  # Use axis=1 for the Y-axis (height)
    
    # Calculate the absolute difference between the mirrored lower half and the upper half
    difference = tf.abs(upper_half - lower_half_mirrored)
    
    # Compute the loss as the mean of the differences
    loss = tf.reduce_mean(difference)
    
    return loss



def normalize_intensities(wm_intensity, gm_intensity, csf_intensity, c):
    total = wm_intensity + gm_intensity + csf_intensity + c

    # Check if the total is below the threshold
    below_threshold = total < matter_th

    # Avoid division by zero
    total = tf.where(below_threshold, tf.constant(1, dtype=dtype), total)
    equal_proportion = tf.constant(0.25, dtype=dtype)

    # Normalize intensities
    normalized_wm = tf.where(below_threshold, equal_proportion, wm_intensity / total)
    normalized_gm = tf.where(below_threshold, equal_proportion, gm_intensity / total)
    normalized_csf = tf.where(below_threshold, equal_proportion, csf_intensity / total)
    normalized_c = tf.where(below_threshold, equal_proportion, c / total)

    return normalized_wm, normalized_gm, normalized_csf, normalized_c


def compute_lambda(wm_intensity, gm_intensity, csf_intensity, c):
    # Normalize the intensities
    wm_intensity, gm_intensity, csf_intensity, c = normalize_intensities(wm_intensity, gm_intensity, csf_intensity, c)
    
    # Constants
    E = [2100, 2100, 100, 8000]  # Young’s modulus for GM, WM, CSF, tumor
    nu = [0.4, 0.4, 0.1, 0.45]  # Poisson’s ratio for GM, WM, CSF, tumor
    
    # Calculating lambda for each material
    lambda_vals = [E[i] * nu[i] / ((1 + nu[i]) * (1 - 2 * nu[i])) for i in range(4)]
    
    # Compute weighted lambda based on normalized intensities
    weighted_lambda = (wm_intensity * lambda_vals[0] + 
                       gm_intensity * lambda_vals[1] + 
                       csf_intensity * lambda_vals[2] + 
                       c * lambda_vals[3])

    return weighted_lambda

def compute_mu(wm_intensity, gm_intensity, csf_intensity, c):
    # Normalize the intensities
    wm_intensity, gm_intensity, csf_intensity, c = normalize_intensities(wm_intensity, gm_intensity, csf_intensity, c)
    
    # Constants
    E = [2100, 2100, 100, 8000]  # Young’s modulus for GM, WM, CSF, tumor
    nu = [0.4, 0.4, 0.1, 0.45]  # Poisson’s ratio for GM, WM, CSF, tumor
    
    # Calculating mu for each material
    mu_vals = [E[i] / (2 * (1 + nu[i])) for i in range(4)]
    
    # Compute weighted mu based on normalized intensities
    weighted_mu = (wm_intensity * mu_vals[0] + 
                   gm_intensity * mu_vals[1] + 
                   csf_intensity * mu_vals[2] + 
                   c * mu_vals[3])

    return weighted_mu


def compute_strain_balance_tf(E, c, gamma, lambda_vals, mu_vals, domain):
    residuals = []  # A list to hold the residuals for each timestep
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')

    for t in range(0, c.shape[0]):
        lambda_ = lambda_vals[t]
        mu = mu_vals[t]
        grad_c_x, grad_c_y, grad_c_z = grad_fd(c[t], dx, dy, dz)
        # Stack gradient to match the dimensionality of div
        tumor_force = gamma * tf.stack([grad_c_x, grad_c_y, grad_c_z], axis=0)  

        # Calculate the total stress
        sigma_tissue =  compute_stress(E[...,t,:,:,:], lambda_, mu)
        div = divergence_fd(sigma_tissue,dx,dy,dz)

        # Calculate the residual
        residual = div + tumor_force

        residuals.append(residual)
        
    return tf.stack(residuals)  # Stack the residuals to form a tensor

def divergence_fd(tensor, dx, dy, dz):
    """
    Compute the divergence of a vector field in 3D using the gradient function.

    Parameters
    ----------
    tensor : tf.Tensor
        The vector field tensor with shape (3,3, nx, ny, nz).
    dx : float, optional
        The step size in the x-direction.
    dy : float, optional
        The step size in the y-direction.
    dz : float, optional
        The step size in the z-direction.

    Returns
    -------
    tf.Tensor
        The divergence of the input tensor.
    """
    # Shift tensor in x, y, and z directions
    tensor_right = tf.roll(tensor, shift=-1, axis=-3)
    tensor_up = tf.roll(tensor, shift=-1, axis=-2)
    tensor_forward = tf.roll(tensor, shift=-1, axis=-1)
    
    # Shift tensor in x, y, and z directions
    tensor_left = tf.roll(tensor, shift=1, axis=-3)
    tensor_down = tf.roll(tensor, shift=1, axis=-2)
    tensor_back = tf.roll(tensor, shift=1, axis=-1)

    # Calculate the derivatives in the x, y, and z directions
    div_x = (tensor_right - tensor_left) / (2*dx)
    div_y = (tensor_up - tensor_down) / (2*dy)
    div_z = (tensor_forward - tensor_back) / (2*dz)

    # Sum the partial derivatives to get the divergence
    div = div_x[...,0,:,:,:] + div_y[...,1,:,:,:] + div_z[...,2,:,:,:]

    return div

    return div

def gradient_3d(tensor, step, axis):
    """
    Compute the gradient of a 3D tensor using a central difference scheme in the interior and a first-order 
    scheme at the boundaries.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to differentiate. Shape: (nx, ny, nz).
    step : float
        The step size.
    axis : int
        The axis along which to compute the gradient.

    Returns
    -------
    tf.Tensor
        The gradient of the input tensor.
    """
    tensor_before = tf.roll(tensor, shift=1, axis=axis)
    tensor_after = tf.roll(tensor, shift=-1, axis=axis)

    # Central difference in the interior of the domain
    gradient = (tensor_after - tensor_before) / (2 * step)

    # Handling boundaries based on the axis
    if axis == 0:
        gradient_left = (tensor[1, :, :] - tensor[0, :, :]) / step
        gradient_right = (tensor[-1, :, :] - tensor[-2, :, :]) / step
        gradient = tf.concat([gradient_left[None, :, :], gradient[1:-1, :, :], gradient_right[None, :, :]], axis=axis)
    
    elif axis == 1:
        gradient_top = (tensor[:, 1, :] - tensor[:, 0, :]) / step
        gradient_bottom = (tensor[:, -1, :] - tensor[:, -2, :]) / step
        gradient = tf.concat([gradient_top[:, None, :], gradient[:, 1:-1, :], gradient_bottom[:, None, :]], axis=axis)

    elif axis == 2:
        gradient_front = (tensor[:, :, 1] - tensor[:, :, 0]) / step
        gradient_back = (tensor[:, :, -1] - tensor[:, :, -2]) / step
        gradient = tf.concat([gradient_front[:, :, None], gradient[:, :, 1:-1], gradient_back[:, :, None]], axis=axis)

    return gradient


def grad_fd(c, dx, dy, dz):
    """
    Compute the gradient of a 3D scalar field using a central difference scheme.

    Parameters
    ----------
    c : tf.Tensor
        The scalar field for which the gradient is to be computed. Shape: (nt, nx, ny, nz).
    dx : float, optional
        The step size in the x-direction.
    dy : float, optional
        The step size in the y-direction.
    dz : float, optional
        The step size in the z-direction.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        The gradients in the x, y, and z directions.
    """
    grad_x = gradient_3d(c, dx, axis=0)  # Compute gradient in x-direction
    grad_y = gradient_3d(c, dy, axis=1)  # Compute gradient in y-direction
    grad_z = gradient_3d(c, dz, axis=2)  # Compute gradient in z-direction

    return grad_x, grad_y, grad_z

def compute_stress(E, lambda_, mu):
    # Compute trace of the strain tensor (sum of diagonal elements)
    trace_E = E[0, 0, ...] + E[1, 1, ...] + E[2, 2, ...]

    # Create identity tensor
    I = tf.eye(3, dtype=dtype)[..., tf.newaxis, tf.newaxis, tf.newaxis] # Shape (3,3,1,1,1)
    I = tf.broadcast_to(I, E.shape)  # Shape (3,3,nx,ny,nz)
    # Calculate the stress tensor components
    sigma = lambda_ * trace_E * I + 2 * mu * E

    return sigma




def tumor_pde_loss(tx, ty, tz, dt, c, D,rho):
    
    # Calculate the field residuals for the PDE
    # First term of the PDE: time derivative of c
    dc_dt = (c - tf.roll(c, shift=1, axis=0)) / dt

    # Reaction term (calculated at the mid-point in time)
    c_mid = 0.5 * (c + tf.roll(c, shift=1, axis=0))
    reaction = tf.abs(c_mid) * (1 - c_mid) * rho

    # Calculate Δx, Δy, and Δz
    dx = (tf.roll(tx, shift=-1, axis=1) - tx)
    dy = (tf.roll(ty, shift=-1, axis=2) - ty)
    dz = (tf.roll(tz, shift=-1, axis=3) - tz)

    # Calculate Δc in x, y, and z directions
    dc_x = tf.roll(c, shift=-1, axis=1) - c
    dc_y = tf.roll(c, shift=-1, axis=2) - c
    dc_z = tf.roll(c, shift=-1, axis=3) - c

    
    diffusion_x = (D["D_minus_x"] * dc_x / dx - D["D_plus_x"]*tf.roll(dc_x / dx, shift=1, axis=1)) / ((dx + tf.roll(dx, shift=1, axis=1)) / 2)
    diffusion_y = (D["D_minus_y"] * dc_y / dy - D["D_plus_y"]*tf.roll(dc_y / dy, shift=1, axis=2)) / ((dy + tf.roll(dy, shift=1, axis=2)) / 2)
    diffusion_z = (D["D_minus_z"] * dc_z / dz - D["D_plus_z"]*tf.roll(dc_z / dz, shift=1, axis=3)) / ((dz + tf.roll(dz, shift=1, axis=3)) / 2)
    
    diffusion = diffusion_x + diffusion_y + diffusion_z
    diffusion = 0.5 * (diffusion + tf.roll(diffusion, shift=1, axis=0))  # Diffusion terms calculated at the mid-point in time

    # Total PDE loss
    pde_loss = dc_dt - reaction - diffusion

    # Exclude the boundary and first two time steps
    return pde_loss[2:, 1:-1, 1:-1, 1:-1]



def initialize_c(nx, ny, nz, x_center, y_center, z_center, radius):
    """
    Initialize a 3D field with a 3D Gaussian distribution.
    
    :param nx: Size in the x-dimension
    :param ny: Size in the y-dimension
    :param nz: Size in the z-dimension
    :param x_center: x-coordinate of the center of the Gaussian distribution
    :param y_center: y-coordinate of the center of the Gaussian distribution
    :param z_center: z-coordinate of the center of the Gaussian distribution
    :param radius: Radius of the Gaussian distribution
    :return: Initialized 3D field
    """
    # Initialize the field to zeros
    c_init = np.zeros((nx, ny, nz))

    # Generate a 3D meshgrid
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')

    # Subtract the center coordinates to shift the Gaussian to the correct location
    x = x - x_center
    y = y - y_center
    z = z - z_center

    # Apply the 3D Gaussian function
    c_init = np.exp(-((x**2 + y**2 + z**2) / (2.0 * radius**2)))

    return c_init.astype(dtype)


def lagrange_to_euler_single_slice(c_lagrange, tx, ty, tz, domain, time_slice_index, method='nearest'):
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')
    
    # Placeholder for the Eulerian field values for a single time slice
    c_eulerian_slice = np.empty((nx, ny, nz))

    # Flatten the trajectories and field arrays for interpolation for the specific time slice.
    tx_flat = tx[time_slice_index].flatten()
    ty_flat = ty[time_slice_index].flatten()
    tz_flat = tz[time_slice_index].flatten()
    c_flat = c_lagrange[time_slice_index].flatten()
    
    # Create a 3D grid for interpolation
    grid_x, grid_y, grid_z = np.mgrid[domain.lower[1]:domain.upper[1]:nx*1j, 
                                      domain.lower[2]:domain.upper[2]:ny*1j, 
                                      domain.lower[3]:domain.upper[3]:nz*1j]
    
    # Perform the interpolation for the specified time slice
    c_eulerian_slice = griddata((tx_flat, ty_flat, tz_flat), c_flat, (grid_x, grid_y, grid_z), method=method)

    # Handle remaining NaNs in the result
    c_eulerian_slice = np.nan_to_num(c_eulerian_slice, nan=0)

    return c_eulerian_slice

from scipy.interpolate import NearestNDInterpolator
def lagrange_to_euler_single_slice2(c_lagrange, tx, ty, tz, domain, time_slice_index):
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Flatten the trajectories and field arrays for interpolation
    points = np.array([tx[time_slice_index].flatten(), 
                       ty[time_slice_index].flatten(), 
                       tz[time_slice_index].flatten()]).T
    values = c_lagrange[time_slice_index].flatten()

    # Create an interpolator for the irregular grid
    interpolator = NearestNDInterpolator(points, values)

    # Create a regular grid for the Eulerian frame
    grid_x, grid_y, grid_z = np.mgrid[domain.lower[1]:domain.upper[1]:nx*1j, 
                                      domain.lower[2]:domain.upper[2]:ny*1j, 
                                      domain.lower[3]:domain.upper[3]:nz*1j]

    # Perform the interpolation onto the regular grid
    c_eulerian_slice = interpolator(grid_x, grid_y, grid_z)

    # Handle remaining NaNs in the result
    c_eulerian_slice = np.nan_to_num(c_eulerian_slice, nan=0)

    return c_eulerian_slice

def lagrange_to_euler_all_slices(c_lagrange, tx, ty, tz, domain):
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')
    nt = c_lagrange.shape[0]  # Assuming the first dimension of c_lagrange is time

    # Placeholder for the Eulerian field values for all time slices
    c_eulerian = np.empty((nt, nx, ny, nz))

    # Create a 3D grid for interpolation
    grid_x, grid_y, grid_z = np.mgrid[domain.lower[1]:domain.upper[1]:nx*1j, 
                                      domain.lower[2]:domain.upper[2]:ny*1j, 
                                      domain.lower[3]:domain.upper[3]:nz*1j]

    for time_slice_index in range(nt):
        # Flatten the trajectories and field arrays for interpolation
        tx_flat = tx[time_slice_index].flatten()
        ty_flat = ty[time_slice_index].flatten()
        tz_flat = tz[time_slice_index].flatten()
        c_flat = c_lagrange[time_slice_index].flatten()

        # Perform the interpolation for each time slice
        c_eulerian_slice = griddata((tx_flat, ty_flat, tz_flat), c_flat, (grid_x, grid_y, grid_z), method='nearest')

        # Handle remaining NaNs in the result
        c_eulerian[time_slice_index] = np.nan_to_num(c_eulerian_slice, nan=0)

    return c_eulerian


def load_image(path):
    '''
    Returns image as np.ndarray.
    Origin is at the bottom left corner.
    '''
    from PIL import Image
    img = Image.open(path)
    u = np.array(img, dtype=float)
    if len(u.shape) == 3 and u.shape[2] in [3, 4]:
        u = np.mean(u[:, :, :3], axis=2)
    elif len(u.shape) == 2:
        pass
    else:
        raise RuntimeError("Unsupported image shape {:}".format(u.shape))
    u = np.fliplr(u.T / 255)
    return u


def save_image(u, path):
    from PIL import Image
    u = np.fliplr((np.clip(u, 0, 1) * 255).astype(np.uint8)).T
    img = Image.fromarray(u)
    img.save(path)
    return img

def plot_and_save(data, path, cmap="gray"):
    plt.figure()  # Start a new figure
    plt.imshow(data.T, cmap=cmap)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the current figure

def get_data(filename):
    """
    Load a 3D volume from either a .npy or .nii(.gz) file.
    Also store the orientation/affine in a global variable if the file is NIfTI.
    """
    global stored_affine
    
    # Split the file name to get the extension
    _, file_extension = os.path.splitext(filename)

    # Handle files with .gz extension (common for NIfTI files)
    if file_extension == '.gz':
        _, file_extension = os.path.splitext(filename[:-3])

    # Load data based on file extension
    if file_extension == '.npy':
        volume = np.load(filename)
        # No orientation info to store for NPY
        stored_affine = None

    elif file_extension == '.nii':
        nii_img = nib.load(filename)
        volume = nii_img.get_fdata()

        # Store affine in the global variable
        stored_affine = nii_img.affine

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return volume


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nt', type=int, default=None, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=65, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=None, help="Grid size in y")
    parser.add_argument('--Nz', type=int, default=None, help="Grid size in z")
    parser.add_argument('--mgloss',
                        type=int,
                        default=0,
                        help="Use multigrid norm in loss")
    parser.add_argument('--kmgloss',
                        type=float,
                        default=0.25,
                        help="Factor for each restriction operator")
    parser.add_argument('--kxreg',
                        type=float,
                        default=0,
                        help="Smoothness of particles in space")
    parser.add_argument('--ktreg',
                        type=float,
                        default=0,
                        help="Smoothness of particles in time")
    parser.add_argument('--kimp',
                        type=float,
                        default=1,
                        help="Imposed values weight")
    parser.add_argument('--karea', type=float, default=0, help="Fix area")
    
    parser.add_argument('--wmfile',
                        type=str,
                        required=True,
                        help="Path to the white matter tissue file")
    parser.add_argument('--gmfile',
                        type=str,
                        required=True,
                        help="Path to the gray matter tissue file")
    
    parser.add_argument('--csffile',
                        type=str,
                        required=True,
                        help="Path to the CSF matter tissue file")
    
    parser.add_argument('--segmfile',
                        type=str,
                        required=True,
                        help="Path to the SEGMENTATION file")
    
    parser.add_argument('--petfile',
                    type=str,
                    required=True,
                    help="Path to the pet file")

    parser.add_argument('--save_full_solution',
                        action='store_true',
                        help="Flag to indicate if the full solution should be saved")
    
    parser.add_argument('--save_last_timestep_solution',
                    action='store_true',
                    help="Flag to indicate if the last timestep solution should be saved")
        
    parser.add_argument('--output_dir',
                    type=str,
                    required=True,
                    help="Out path")

    

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(dump_data=1)
    parser.set_defaults(multigrid=1)
    #parser.set_defaults(mg_interp='rollstack')
    parser.set_defaults(plot_every=5000, report_every=1000, history_every=14999)
    parser.set_defaults(history_full=1)
    parser.set_defaults(optimizer='adamn')
    #parser.set_defaults(optimizer='lbfgs')

    parser.set_defaults(every_factor=1)
    parser.set_defaults(lr=0.001)
    parser.set_defaults(frames=5)
    return parser.parse_args()


def state_to_traj(domain, state, mod=np):
    _, x, y, z = domain.points()
    tx = domain.field(state, 'x')
    ty = domain.field(state, 'y')
    tz = domain.field(state, 'z')
    tx, ty, tz = transform_txyz(tx, ty, tz, x, y, z, mod=mod)
    return tx, ty, tz



def particles_to_field_3d_average(up, tx, ty, tz, domain):
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Clip the trajectories within the domain boundaries
    tx = tf.clip_by_value(tx, domain.lower[1], domain.upper[1])
    ty = tf.clip_by_value(ty, domain.lower[2], domain.upper[2])
    tz = tf.clip_by_value(tz, domain.lower[3], domain.upper[3])

    # Calculate nearest grid indices
    ix = tf.cast(tf.round((tx - domain.lower[1]) / dx), tf.int32)
    iy = tf.cast(tf.round((ty - domain.lower[2]) / dy), tf.int32)
    iz = tf.cast(tf.round((tz - domain.lower[3]) / dz), tf.int32)

    # Ensure indices are within bounds
    ix = tf.clip_by_value(ix, 0, nx - 1)
    iy = tf.clip_by_value(iy, 0, ny - 1)
    iz = tf.clip_by_value(iz, 0, nz - 1)

    # Flatten the indices to 1D to simplify aggregation
    flat_indices = tf.stack([ix, iy, iz], axis=-1)

    # Initialize the output grid
    u = tf.zeros((nx, ny, nz), dtype=up.dtype)

    # Handling multiple particles mapped to the same grid point by averaging their contributions
    # Step 1: Create a tensor of ones to use for counting particle contributions per grid cell
    ones = tf.ones_like(up)

    # Step 2: Aggregate contributions and counts
    u = tf.tensor_scatter_nd_add(u, flat_indices, up)
    counts = tf.tensor_scatter_nd_add(tf.zeros((nx, ny, nz), dtype=up.dtype), flat_indices, ones)

    # Avoid division by zero by setting zero counts to one (or handle differently as needed)
    counts = tf.where(counts == 0, tf.ones_like(counts), counts)

    # Step 3: Compute the average by dividing the sum of contributions by the count
    u_average = u / counts

    return u_average


def particles_to_field_3d(up, tx, ty, tz, domain):
    '''
    up: values carried by particles, shape (nx, ny, nz)
    tx, ty, tz: trajectories of particles, shape (nx, ny, nz) as TensorFlow tensors
    '''
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Clip the trajectories within the domain boundaries
    tx = tf.clip_by_value(tx, domain.lower[1], domain.upper[1])
    ty = tf.clip_by_value(ty, domain.lower[2], domain.upper[2])
    tz = tf.clip_by_value(tz, domain.lower[3], domain.upper[3])

    # Calculate indices for x, y, and z dimensions
    tix = tf.clip_by_value(tf.math.floor(tx / dx - 0.5), 0, nx - 2)
    tiy = tf.clip_by_value(tf.math.floor(ty / dy - 0.5), 0, ny - 2)
    tiz = tf.clip_by_value(tf.math.floor(tz / dz - 0.5), 0, nz - 2)
    tix = tf.cast(tix, tf.int32)
    tiy = tf.cast(tiy, tf.int32)
    tiz = tf.cast(tiz, tf.int32)
    tixp = tix + 1
    tiyp = tiy + 1
    tizp = tiz + 1


    # Calculate weights for x, y, and z dimensions
    sx1 = tf.clip_by_value(tx / dx - 0.5 - tf.cast(tix, tf.float32), 0., 1.)
    sy1 = tf.clip_by_value(ty / dy - 0.5 - tf.cast(tiy, tf.float32), 0., 1.)
    sz1 = tf.clip_by_value(tz / dz - 0.5 - tf.cast(tiz, tf.float32), 0., 1.)
    sx0 = 1 - sx1
    sy0 = 1 - sy1
    sz0 = 1 - sz1

    # Flatten the particle values to match the flat index tensor
    up = tf.reshape(up, [-1])

    # Initialize the output field and weight tensors
    u = tf.zeros_like(tx, dtype=domain.dtype)
    w = tf.zeros_like(tx, dtype=domain.dtype)

    # Generate indices for x, y, z dimensions
    ix, iy, iz = tf.meshgrid(tf.range(nx), tf.range(ny), tf.range(nz), indexing='ij')
    flat_indices = tf.stack([ix, iy, iz], axis=-1)
    flat_indices = tf.reshape(flat_indices, [-1, 3])

    # Iterate over all combinations of indices and weights
    for jx, jy, jz, jw in [
        (tix, tiy, tiz, sx0 * sy0 * sz0),
        (tixp, tiy, tiz, sx1 * sy0 * sz0),
        (tix, tiyp, tiz, sx0 * sy1 * sz0),
        (tixp, tiyp, tiz, sx1 * sy1 * sz0),
        (tix, tiy, tizp, sx0 * sy0 * sz1),
        (tixp, tiy, tizp, sx1 * sy0 * sz1),
        (tix, tiyp, tizp, sx0 * sy1 * sz1),
        (tixp, tiyp, tizp, sx1 * sy1 * sz1),
    ]:
        combined_indices = tf.stack([tf.reshape(jx, [-1]), tf.reshape(jy, [-1]), tf.reshape(jz, [-1])], axis=1)
        combined_weights = tf.reshape(jw, [-1])
        up_weights = up * combined_weights

    u = tf.tensor_scatter_nd_add(u, combined_indices, up_weights)
    w = tf.tensor_scatter_nd_add(w, combined_indices, combined_weights)


    # Normalize the field values by the weights
    u = u / (w + 1e-8)
    return u


def particles_to_field(up, tx, ty, tz, domain):
    '''
    up: values carried by particles, shape (nx, ny, nz)
    tx, ty, tz: trajectories of particles, shape (nt, nx, ny, nz)
    '''
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Clip the trajectories within the domain boundaries
    tx = np.clip(tx, domain.lower[1], domain.upper[1])
    ty = np.clip(ty, domain.lower[2], domain.upper[2])
    tz = np.clip(tz, domain.lower[3], domain.upper[3])

    # Calculate indices for x, y, and z dimensions
    tix = np.clip(np.floor(tx / dx - 0.5), 0, nx - 2).astype(np.int32)
    tiy = np.clip(np.floor(ty / dy - 0.5), 0, ny - 2).astype(np.int32)
    tiz = np.clip(np.floor(tz / dz - 0.5), 0, nz - 2).astype(np.int32)
    tixp = tix + 1
    tiyp = tiy + 1
    tizp = tiz + 1

    # Calculate weights for x, y, and z dimensions
    sx1 = np.clip(tx / dx - 0.5 - tix, 0., 1.)
    sy1 = np.clip(ty / dy - 0.5 - tiy, 0., 1.)
    sz1 = np.clip(tz / dz - 0.5 - tiz, 0., 1.)
    sx0 = 1 - sx1
    sy0 = 1 - sy1
    sz0 = 1 - sz1

    # Broadcast and flatten the particle values
    up = up[None, ...]
    up = np.broadcast_to(up, tx.shape).flatten()
    u = np.zeros(tx.shape, dtype=domain.dtype)
    w = np.zeros(tx.shape, dtype=domain.dtype)
    it,ix,iy,iz = domain.indices()
    it = np.array(it).flatten()
    # Iterate over all combinations of indices and weights
    for jx, jy, jz, jw in [
        (tix, tiy, tiz, sx0 * sy0 * sz0),
        (tixp, tiy, tiz, sx1 * sy0 * sz0),
        (tix, tiyp, tiz, sx0 * sy1 * sz0),
        (tixp, tiyp, tiz, sx1 * sy1 * sz0),
        # Add combinations for z-dimension
        (tix, tiy, tizp, sx0 * sy0 * sz1),
        (tixp, tiy, tizp, sx1 * sy0 * sz1),
        (tix, tiyp, tizp, sx0 * sy1 * sz1),
        (tixp, tiyp, tizp, sx1 * sy1 * sz1),
    ]:
        jx = jx.flatten()
        jy = jy.flatten()
        jz = jz.flatten()
        jw = jw.flatten()
        np.add.at(u, (it, jx, jy, jz), up * jw)
        np.add.at(w, (it, jx, jy, jz), jw)

    # Normalize the field values by the weights
    u = u / (w + 1e-8)
    return u


def particles_to_field_tf(up, tx, ty, tz, domain):
    '''
    up: values carried by particles, shape (nx, ny, nz)
    tx, ty, tz: trajectories of particles, shape (nt, nx, ny, nz) as TensorFlow tensors
    '''
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Clip the trajectories within the domain boundaries
    tx = tf.clip_by_value(tx, domain.lower[1], domain.upper[1])
    ty = tf.clip_by_value(ty, domain.lower[2], domain.upper[2])
    tz = tf.clip_by_value(tz, domain.lower[3], domain.upper[3])

    # Calculate indices for x, y, and z dimensions
    tix = tf.clip_by_value(tf.math.floor(tx / dx - 0.5), 0, nx - 2)
    tiy = tf.clip_by_value(tf.math.floor(ty / dy - 0.5), 0, ny - 2)
    tiz = tf.clip_by_value(tf.math.floor(tz / dz - 0.5), 0, nz - 2)
    tix = tf.cast(tix, tf.int32)
    tiy = tf.cast(tiy, tf.int32)
    tiz = tf.cast(tiz, tf.int32)
    tixp = tix + 1
    tiyp = tiy + 1
    tizp = tiz + 1

    # Calculate weights for x, y, and z dimensions
    sx1 = tf.clip_by_value(tx / dx - 0.5 - tf.cast(tix, tf.float32), 0., 1.)
    sy1 = tf.clip_by_value(ty / dy - 0.5 - tf.cast(tiy, tf.float32), 0., 1.)
    sz1 = tf.clip_by_value(tz / dz - 0.5 - tf.cast(tiz, tf.float32), 0., 1.)
    sx0 = 1 - sx1
    sy0 = 1 - sy1
    sz0 = 1 - sz1

    # Broadcast and flatten the particle values
    up = tf.reshape(up, [1] + up.shape.as_list())
    up = tf.broadcast_to(up, tx.shape)
    up = tf.reshape(up, [-1])

    u = tf.Variable(tf.zeros_like(tx, dtype=domain.dtype))
    w = tf.Variable(tf.zeros_like(tx, dtype=domain.dtype))

    # Generate indices
    shape = tx.shape.as_list()
    it, ix, iy, iz = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), tf.range(shape[3]), indexing='ij')
    flat_indices = tf.stack([it, ix, iy, iz], axis=-1)
    flat_indices = tf.reshape(flat_indices, [-1, 4])

    # Iterate over all combinations of indices and weights
    for jx, jy, jz, jw in [
        (tix, tiy, tiz, sx0 * sy0 * sz0),
        (tixp, tiy, tiz, sx1 * sy0 * sz0),
        (tix, tiyp, tiz, sx0 * sy1 * sz0),
        (tixp, tiyp, tiz, sx1 * sy1 * sz0),
        (tix, tiy, tizp, sx0 * sy0 * sz1),
        (tixp, tiy, tizp, sx1 * sy0 * sz1),
        (tix, tiyp, tizp, sx0 * sy1 * sz1),
        (tixp, tiyp, tizp, sx1 * sy1 * sz1),
    ]:
        combined_indices = tf.stack([tf.reshape(jx, [-1]), tf.reshape(jy, [-1]), tf.reshape(jz, [-1]), tf.reshape(flat_indices[:, 0], [-1])], axis=1)
        combined_weights = tf.reshape(jw, [-1])
        up_weights = up * combined_weights

        u.assign(tf.tensor_scatter_nd_add(u, combined_indices, up_weights))
        w.assign(tf.tensor_scatter_nd_add(w, combined_indices, combined_weights))

    # Normalize the field values by the weights
    u = u / (w + 1e-8)
    return u


def pad_linear(u, paddings):
    ur = np.pad(u, paddings, mode='reflect')
    us = np.pad(u, paddings, mode='symmetric')
    u = 2 * us - ur
    return u

def field_to_particles(u_src, tx, ty, tz, domain):
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Offset from corner cell center in x, y, and z dimensions.
    dtx = tx / dx - 0.5
    dty = ty / dy - 0.5
    dtz = tz / dz - 0.5

    # Indices for x, y, and z dimensions.
    tix = np.clip(np.floor(dtx).astype(np.int32), -1, nx - 1)
    tiy = np.clip(np.floor(dty).astype(np.int32), -1, ny - 1)
    tiz = np.clip(np.floor(dtz).astype(np.int32), -1, nz - 1)
    tixp = tix + 1
    tiyp = tiy + 1
    tizp = tiz + 1

    # Weights for x, y, and z dimensions.
    sx1 = np.clip(dtx - tix, 0, 1)
    sy1 = np.clip(dty - tiy, 0, 1)
    sz1 = np.clip(dtz - tiz, 0, 1)
    sx0 = np.clip(tixp - dtx, 0, 1)
    sy0 = np.clip(tiyp - dty, 0, 1)
    sz0 = np.clip(tizp - dtz, 0, 1)

    # Pad the source field to handle boundary conditions
    u_src = pad_linear(u_src, [(1, 1), (1, 1), (1, 1)])

    # Compute the resulting field
    res = np.zeros(tx.shape, dtype=domain.dtype)
    for jx, jy, jz, jw in [
        (tix, tiy, tiz, sx0 * sy0 * sz0),
        (tixp, tiy, tiz, sx1 * sy0 * sz0),
        (tix, tiyp, tiz, sx0 * sy1 * sz0),
        (tixp, tiyp, tiz, sx1 * sy1 * sz0),
        # Add combinations for z-dimension
        (tix, tiy, tizp, sx0 * sy0 * sz1),
        (tixp, tiy, tizp, sx1 * sy0 * sz1),
        (tix, tiyp, tizp, sx0 * sy1 * sz1),
        (tixp, tiyp, tizp, sx1 * sy1 * sz1),
    ]:
        res += jw * u_src[tix + 1, tiy + 1, tiz + 1]
    return res

def report_func(problem, state, epoch, cbinfo):
    global segm_data
    domain = problem.domain
    coeff = np.array(domain.field(state, 'coeff'))
    
    tx, ty, tz = state_to_traj(domain, state, mod=np)
    tx, ty, tz = np.array(tx), np.array(ty), np.array(tz)
    c_lagrange = np.array(transform_c((domain.field(state, 'c')),tf))
    c_euler_last_slice = lagrange_to_euler_single_slice(c_lagrange, tx, ty, tz, domain, -1)
    
    edema_dice, core_dice = calculate_dice_scores(segm_data, coeff, c_euler_last_slice)
    # Print current parameters.
    printlog('coeff={}'.format(coeff))
    printlog('dice={}'.format( [np.array(edema_dice), np.array(core_dice)]))


def history_func(problem, state, epoch, history, cbinfo):
    
    if args.save_full_solution and epoch > 0:
        process_and_save_solution(problem, state, wm_data, gm_data, csf_data, epoch)

def process_and_save_solution(problem, state, wm_data, gm_data, csf_data, epoch):
    domain = problem.domain
    mod = domain.mod

    # Time slices.
    tx, ty, tz = state_to_traj(domain, state, mod=mod)
    tx, ty, tz = np.array(tx), np.array(ty), np.array(tz)

    # Initialize empty lists to store the data for each tissue type
    wm_results = []
    gm_results = []
    csf_results = []

    wm_mids = field_to_particles(wm_data, tx[-1], ty[-1], tz[-1], domain)
    wm_results = particles_to_field(wm_mids, tx, ty, tz, domain)

    gm_mids = field_to_particles(gm_data, tx[-1], ty[-1], tz[-1], domain)
    gm_results = particles_to_field(gm_mids, tx, ty, tz, domain)

    csf_mids = field_to_particles(csf_data, tx[-1], ty[-1], tz[-1], domain)
    csf_results = particles_to_field(csf_mids, tx, ty, tz, domain)

    c_lagrange = np.array(transform_c(domain.field(state, 'c'), mod))
    c_euler = np.array(lagrange_to_euler_all_slices(c_lagrange, tx, ty, tz, domain))

    data_dict = {
        'wm_data': restore_all_timepoints(wm_results,10), #save every 10nt
        'gm_data': restore_all_timepoints(gm_results,10),
        'csf_data': restore_all_timepoints(csf_results,10),
        'c_euler': restore_all_timepoints(c_euler,10),
        'tx': restore_all_timepoints(tx,10),
        'ty': restore_all_timepoints(ty,10),
        'tz': restore_all_timepoints(tz,10),
        
    }

    filename = f'tissue_data4D_epoch{epoch}.npy'
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, protocol=4)
    
    print(f"Solution saved for epoch {epoch}.")


def plot(problem, state, epoch, frame, cbinfo=None):
    global wm_data, gm_data, csf_data, outside_skull_mask, segm_data
    domain = problem.domain
    mod = domain.mod
    path = f"{frame}_cmp_3d"

    __, xx, yy, zz = domain.points()
    nx = domain.cshape[1]
    ny = domain.cshape[2]
    nz = domain.cshape[3]
    dt = domain.upper[0] / domain.cshape[0]
    dx = domain.upper[1] / domain.cshape[1]
    dy = domain.upper[2] / domain.cshape[2]
    dz = domain.upper[3] / domain.cshape[3]

    tx, ty, tz = state_to_traj(domain, state, mod=mod)
    tx, ty, tz = np.array(tx), np.array(ty), np.array(tz)
    wm_mids = field_to_particles(wm_data, tx[-1], ty[-1], tz[-1], domain)
    wm_intensities = particles_to_field(wm_mids, tx, ty, tz, domain)
    c_lagrange = np.array(transform_c(domain.field(state, 'c'),mod))
    # Select a middle slice for visualization
    CM_pos = center_of_mass(np.where(segm_data == 4, 1,0))
    middle_z = int(CM_pos[2])
    fig, axes = plt.subplots(7, 7, figsize=(14, 14))
    extent = [domain.lower[1], domain.upper[1], domain.lower[2], domain.upper[2]]
    timepoints = np.linspace(1, tx.shape[0] - 1, 6, dtype=int)
    timepoints = np.insert(timepoints, 0, 0)

    x_min = domain.lower[1]
    y_min = domain.lower[2]
    x_max = domain.upper[1]
    y_max = domain.upper[2]
    
    # Ensuring th_down is no less than 0.20 and no more than 0.35
    th_down = tf.clip_by_value(domain.field(state, 'coeff')[5], clip_value_min=0.20, clip_value_max=0.35)

    # Clip th_up values
    # Ensuring th_up is no less than 0.50 and no more than 0.85
    th_up = tf.clip_by_value(domain.field(state, 'coeff')[6], clip_value_min=0.50, clip_value_max=0.85)

    # Calculate the strain tensor
    ux, uy, uz = compute_displacement_np(tx, ty, tz)
    E = compute_strain_tensor_lagrangian_full_np(ux, uy, uz, dx, dy, dz)
    # Calculate the magnitude of the strain tensor at each time point
    magnitude = np.sqrt(E[0, 0]**2 + E[0, 1]**2 + E[0, 2]**2 + 
                    E[1, 0]**2 + E[1, 1]**2 + E[1, 2]**2 +
                    E[2, 0]**2 + E[2, 1]**2 + E[2, 2]**2)
    
    for i, t in enumerate(timepoints):
        axes[0, i].scatter(tx[t, :, :, middle_z].flatten(),
                           ty[t, :, :, middle_z].flatten(),
                           marker='o',
                           edgecolor='none',
                           facecolor='r',
                           s=1.2,
                           zorder=3,
                           alpha=0.5)
        axes[0, i].set_axis_off()
        axes[0, i].set_xlim([x_min, x_max])  # Set x limits
        axes[0, i].set_ylim([y_min, y_max])  # Set y limits
        axes[0, i].set_title('Trajectories t={:.2f}'.format(t * dt), fontsize=10)

        # Compute Eulerian representation for the current time slice
        c_euler_slice = lagrange_to_euler_single_slice(c_lagrange, tx, ty, tz, domain, t)
        magnitude_slice = lagrange_to_euler_single_slice(magnitude, tx, ty, tz, domain, t)
        vmin = np.min(magnitude_slice)
        vmax = np.max(magnitude_slice)
        
        axes[1, i].imshow(np.where(pet_data > 0,1,np.nan)[:, :, middle_z].T,
                          cmap='Reds',
                          extent=extent,
                          origin='lower',
                          aspect='equal',
                          vmin=0, vmax=1)
        axes[1, i].set_title('pet', fontsize=10)
        axes[1, i].set_xlim([x_min, x_max])
        axes[1, i].set_ylim([y_min, y_max])
        axes[1, i].set_axis_off()
        
        axes[2, i].imshow(np.where(get_core_mask(segm_data),1,np.nan)[:, :, middle_z].T,
                          cmap='Reds',
                          extent=extent,
                          origin='lower',
                          aspect='equal',
                          vmin=0, vmax=1)
        axes[2, i].set_title('Seg', fontsize=10)
        axes[2, i].set_xlim([x_min, x_max])
        axes[2, i].set_ylim([y_min, y_max])
        axes[2, i].set_axis_off()
        
        axes[2, i].imshow(np.where(get_edema_mask(segm_data),1,np.nan)[:, :, middle_z].T,
                          cmap='Greens',
                          extent=extent,
                          origin='lower',
                          aspect='equal',
                          vmin=0, vmax=1)
        axes[2, i].set_xlim([x_min, x_max])  
        axes[2, i].set_ylim([y_min, y_max])
        axes[2, i].set_axis_off()
        
        axes[3, i].imshow(np.where(c_euler_slice[:, :, middle_z] > th_up,1,np.nan).T,
                          cmap='Reds',
                          extent=extent,
                          origin='lower',
                          aspect='equal',
                          vmin=0, vmax=1)
        axes[3, i].set_title('Proposed seg', fontsize=10)
        axes[3, i].set_xlim([x_min, x_max])
        axes[3, i].set_ylim([y_min, y_max])
        axes[3, i].set_axis_off()
        
        axes[3, i].imshow(np.where(np.logical_and(c_euler_slice[:, :, middle_z] < th_up, c_euler_slice[:, :, middle_z] > th_down),1,np.nan).T,
                          cmap='Greens',
                          extent=extent,
                          origin='lower',
                          aspect='equal',
                          vmin=0, vmax=1)
        axes[3, i].set_xlim([x_min, x_max])  
        axes[3, i].set_ylim([y_min, y_max])
        axes[3, i].set_axis_off()
        

        
        axes[4, i].imshow(c_euler_slice[:, :, middle_z].T,
                          interpolation='nearest',
                          cmap='gray',
                          extent=extent,
                          origin='lower',
                          aspect='equal',
                          vmin=0, vmax=1)
        axes[4, i].set_title('C Field', fontsize=10)
        axes[4, i].set_xlim([x_min, x_max])  # Set x limits
        axes[4, i].set_ylim([y_min, y_max])  # Set y limits
        axes[4, i].set_axis_off()
        
        # Add the white matter intensity plot
        wm_im = axes[5, i].imshow(wm_intensities[t, :, :, middle_z].T,
                                interpolation='nearest',
                                cmap='gray',  # or choose another colormap that suits your data
                                extent=extent,
                                origin='lower',
                                aspect='equal')
        axes[5, i].set_title('WHITE MATTER', fontsize=10)
        axes[5, i].set_xlim([x_min, x_max])  # Set x limits
        axes[5, i].set_ylim([y_min, y_max])  # Set y limits
        axes[5, i].set_axis_off()
    
        # Display the magnitude of the strain tensor
        axes[6, i].imshow(magnitude_slice[:, :, middle_z].T,
                        interpolation='nearest',
                        cmap='Greys',
                        extent=extent,
                        origin='lower',
                        vmin=vmin, vmax=vmax)  # use vmin and vmax here
        # Determine the order of magnitude of the min and max magnitudes
        log_max = np.log10(np.max(magnitude[t, :, :]))
        # Display these as text on the plot
        axes[6, i].text(x_min, y_max, "Max: ~10^{:.1f}".format(log_max), fontsize=15, color='red', ha='left', va='top')
        axes[6, i].set_axis_off()
        axes[6, i].set_xlim([x_min, x_max])  # Set x limits
        axes[6, i].set_ylim([y_min, y_max])  # Set y limits
        axes[6, i].set_axis_off()


    fig.subplots_adjust(hspace=0.005, wspace=0.005)
    fig.tight_layout()
    plt.savefig(path, pad_inches=0.01, transparent=False, dpi=300)
    return 0

def plot_final(problem, state):
    pass


# Global variables for storing crop bounds, zoom factors, and original shape
global crop_bounds
global zoom_factors
global original_shape

def process_data(args, matter_th):
    global crop_bounds, zoom_factors, original_shape, outside_matter_mask
    nx, ny, nz = args.Nx, args.Ny, args.Nz

    wm_data = get_data(args.wmfile)
    gm_data = get_data(args.gmfile)
    csf_data = get_data(args.csffile)
    segm_data = get_data(args.segmfile).astype(int)
    pet_data = get_data(args.petfile)

    # Check data types
    print("Initial Data Types:")
    print(f"WM: {wm_data.dtype}, GM: {gm_data.dtype}, CSF: {csf_data.dtype}")

    # Convert to float if necessary
    if not np.issubdtype(wm_data.dtype, np.floating):
        wm_data = wm_data.astype(float)
    if not np.issubdtype(gm_data.dtype, np.floating):
        gm_data = gm_data.astype(float)
    if not np.issubdtype(csf_data.dtype, np.floating):
        csf_data = csf_data.astype(float)

    # Thresholding WM, GM, and CSF data at 5%
    threshold_value = 0.03  # Adjust based on data scale
    wm_data_thresholded = wm_data.copy()
    gm_data_thresholded = gm_data.copy()
    csf_data_thresholded = csf_data.copy()

    wm_data_thresholded[wm_data_thresholded < threshold_value] = 0
    gm_data_thresholded[gm_data_thresholded < threshold_value] = 0
    csf_data_thresholded[csf_data_thresholded < threshold_value] = 0

    print("After Thresholding:")
    print("WM unique values:", np.unique(wm_data_thresholded))
    print("GM unique values:", np.unique(gm_data_thresholded))
    print("CSF unique values:", np.unique(csf_data_thresholded))

    # Update data variables
    wm_data = wm_data_thresholded
    gm_data = gm_data_thresholded
    csf_data = csf_data_thresholded

    print("Resolution of WM data before cropping:", wm_data.shape)
    print("Resolution of GM data before cropping:", gm_data.shape)
    print("Resolution of CSF data before cropping:", csf_data.shape)
    original_shape = wm_data.shape[:3]  # Store original shape globally

    combined_data = wm_data + gm_data + csf_data
    print("Combined data range after thresholding:", combined_data.min(), combined_data.max())

    initial_mask = (combined_data < matter_th).astype(int)
    print("Initial mask unique values:", np.unique(initial_mask))
    print(f"Number of voxels initially below threshold: {np.sum(initial_mask)}")

    def find_crop_boundaries(mask, border):
        border = int(np.ceil(border))
        x_nonzero, y_nonzero, z_nonzero = np.where(mask == 0)
        if x_nonzero.size == 0 or y_nonzero.size == 0 or z_nonzero.size == 0:
            raise ValueError("Mask has no non-zero regions to define cropping boundaries.")
        x_min = max(x_nonzero.min() - border, 0)
        x_max = min(x_nonzero.max() + border, mask.shape[0])
        y_min = max(y_nonzero.min() - border, 0)
        y_max = min(y_nonzero.max() + border, mask.shape[1])
        z_min = max(z_nonzero.min() - border, 0)
        z_max = min(z_nonzero.max() + border, mask.shape[2])
        return x_min, x_max, y_min, y_max, z_min, z_max

    # Find cropping boundaries
    try:
        x_min, x_max, y_min, y_max, z_min, z_max = find_crop_boundaries(initial_mask, 2)
        crop_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)  # Store globally
        print("Crop bounds:", crop_bounds)
    except ValueError as e:
        print("Error finding crop boundaries:", e)
        return None  # Or handle appropriately

    wm_data_cropped = wm_data[x_min:x_max, y_min:y_max, z_min:z_max]
    gm_data_cropped = gm_data[x_min:x_max, y_min:y_max, z_min:z_max]
    csf_data_cropped = csf_data[x_min:x_max, y_min:y_max, z_min:z_max]
    segm_data_cropped = segm_data[x_min:x_max, y_min:y_max, z_min:z_max]
    pet_data_cropped = pet_data[x_min:x_max, y_min:y_max, z_min:z_max]

    print("Resolution of WM data after cropping:", wm_data_cropped.shape)
    print("Resolution of GM data after cropping:", gm_data_cropped.shape)
    print("Resolution of CSF data after cropping:", csf_data_cropped.shape)
    print("Resolution of PET data after cropping:", pet_data_cropped.shape)

    wm_data = wm_data_cropped
    gm_data = gm_data_cropped
    csf_data = csf_data_cropped
    segm_data = segm_data_cropped
    pet_data = pet_data_cropped

    # Calculate zoom factors
    zoom_factors = (
        nx / wm_data_cropped.shape[0],
        ny / wm_data_cropped.shape[1],
        nz / wm_data_cropped.shape[2],
    )
    print("Zoom factors:", zoom_factors)

    # Zoom data to match original dimensions
    wm_data = zoom(wm_data_cropped, zoom_factors, order=1)
    gm_data = zoom(gm_data_cropped, zoom_factors, order=1)
    csf_data = zoom(csf_data_cropped, zoom_factors, order=1)
    segm_data = zoom(segm_data_cropped, zoom_factors, order=0).astype(int)  # Use nearest-neighbor for labels
    pet_data = zoom(pet_data_cropped, zoom_factors, order=1)

    # Get the tumor core mask
    tumor_core_mask = get_core_mask(segm_data, mod=np)  # Using numpy here as we're working with numpy arrays

    # Move CSF within tumor core mask to GM
    gm_data[tumor_core_mask] += csf_data[tumor_core_mask]
    csf_data[tumor_core_mask] = 0

    combined_data = wm_data + gm_data + csf_data
    print("Combined data range after zooming and CSF adjustment:", combined_data.min(), combined_data.max())

    outside_skull_mask = (combined_data < matter_th).astype(int)
    outside_skull_mask = correct_outside_skull_mask(outside_skull_mask)
    print("Outside skull mask unique values:", np.unique(outside_skull_mask))
    print(f"Number of voxels outside skull: {np.sum(outside_skull_mask)}")

    outside_matter_mask = (wm_data + gm_data < matter_th).astype(int)
    print("Outside matter mask unique values:", np.unique(outside_matter_mask))

    CM_pos = center_of_mass(np.where(segm_data == 4, 1, 0))
    middle_z = int(CM_pos[2])
    print(f"Center of mass position: {CM_pos}, middle_z: {middle_z}")

    # Clipping the data
    wm_data = np.clip(wm_data, 0, 1)
    gm_data = np.clip(gm_data, 0, 1)
    csf_data = np.clip(csf_data, 0, 1)

    # Normalizing the data inside the mask
    total_data = wm_data + gm_data + csf_data
    mask_indices = np.where(outside_skull_mask == 0)

    # Avoid division by zero
    total_data_safe = total_data.copy()
    total_data_safe[total_data_safe == 0] = 1

    wm_data[mask_indices] /= total_data_safe[mask_indices]
    gm_data[mask_indices] /= total_data_safe[mask_indices]
    csf_data[mask_indices] /= total_data_safe[mask_indices]

    # Setting data to zero outside the skull
    outside_mask_indices = np.where(outside_skull_mask == 1)
    wm_data[outside_mask_indices] = 0
    gm_data[outside_mask_indices] = 0
    csf_data[outside_mask_indices] = 0

    # Plotting (assuming plot_and_save is correctly defined elsewhere)
    plot_and_save(wm_data[:, :, middle_z], 'white_matter_middle_slice_plot.png')
    plot_and_save(gm_data[:, :, middle_z], 'gray_matter_middle_slice_plot.png')
    plot_and_save(csf_data[:, :, middle_z], 'csf_matter_middle_slice_plot.png')
    plot_and_save(outside_skull_mask[:, :, middle_z], 'outside_skull_mask_matter_middle_slice_plot.png')
    plot_and_save(segm_data[:, :, middle_z], 'segm_middle_slice_plot.png')
    plot_and_save(pet_data[:, :, middle_z], 'pet_middle_slice_plot.png')

    # Saving processed data
    np.save('csf_processed.npy', csf_data)
    np.save('wm_processed.npy', wm_data)
    np.save('gm_processed.npy', gm_data)
    np.save('segm_processed.npy', segm_data)
    np.save('pet_processed.npy', pet_data)

    return wm_data, gm_data, csf_data, segm_data, pet_data, outside_skull_mask, CM_pos


def restore(wm_data):
    global crop_bounds, zoom_factors, original_shape
    # Use the global variables to restore the data
    restored_data = np.zeros(original_shape, dtype=wm_data.dtype)  # Create an empty array of the original shape
    x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds

    # Reverse zoom
    wm_data_zoomed = zoom(wm_data, (1/zoom_factors[0], 1/zoom_factors[1], 1/zoom_factors[2]), order=1)

    # Insert the data back into the correct position
    restored_data[x_min:x_max, y_min:y_max, z_min:z_max] = wm_data_zoomed

    return restored_data

def restore_all_timepoints(data_4D, n):
    global crop_bounds, zoom_factors, original_shape
    num_timepoints = data_4D.shape[0]
    
    # Determine which timepoints to process
    indices_to_process = set(range(0, num_timepoints, n))  # Every nth timepoint
    indices_to_process.add(0)  # Ensure the first timepoint is included
    indices_to_process.add(num_timepoints - 1)  # Ensure the last timepoint is included
    indices_to_process = sorted(indices_to_process)  # Sort the indices
    
    # Initialize an empty 4D array with a size based on the number of timepoints to process
    restored_data_4D = np.zeros((len(indices_to_process),) + original_shape, dtype=data_4D.dtype)
    
    for i, t in enumerate(indices_to_process):
        # Apply the `restore` function to each selected time slice
        restored_data_4D[i] = restore(data_4D[t])

    return restored_data_4D



def make_problem(args):
    global wm_data, gm_data, csf_data, outside_skull_mask, segm_data,pet_data, dtype, CM_pos, matter_th, TS, gamma_ch
    wm_data, gm_data, csf_data, segm_data, pet_data, outside_skull_mask, CM_pos = process_data(args, matter_th)

    dtype = np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx, args.Ny, args.Nz),
                         dimnames=('t', 'x', 'y', 'z'),
                         lower=(0, 0, 0, 0),
                         upper=(100, wm_data.shape[0],wm_data.shape[1],wm_data.shape[2]), #assuming 100 days, 1mm resolution
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_nlvl=args.nlvl)




    if domain.multigrid:
        printlog('multigrid levels:', domain.mg_cshapes)

    tt, xx, yy, zz = domain.points()
    t1 = domain.points_1d('t', loc='c')
    x1 = domain.points_1d('x', loc='c')
    y1 = domain.points_1d('y', loc='c')
    z1 = domain.points_1d('z', loc='c')

    x2 = xx[0]
    y2 = yy[0]
    z2 = zz[0]
    op = partial(operator_adv)
    problem = odil.Problem(op, domain)


    
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')
    nt = domain.size('t')
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    
    parameters = {
    'Dw': D_ch,         # Diffusion coefficient for white matter
    'rho': rho_ch,        # Proliferation rate
    'RatioDw_Dg': R_ch,  # Ratio of diffusion coefficients in white and grey matter
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': CM_pos[0]/nx,    # initial focal position (in percentages)
    'NyT1_pct': CM_pos[1]/ny,
    'NzT1_pct': CM_pos[2]/nz,
    'resolution_factor': 1,
    'time_series_solution_Nt': nt-1,
    'dx_mm': dx,
    'dy_mm': dy,
    'dz_mm': dz,
    'init_scale': 0.8
    }
    sol = Solver(parameters)
    result = sol.solve()
    FK_ts = np.array(result['time_series']).astype(dtype)
    
    # Create an array of zeros with the same shape as one time step
    initial_zero_state = np.zeros(FK_ts[0].shape)
    # Prepend the initial zero state to the time series data
    FK_ts = np.vstack([initial_zero_state[np.newaxis, ...], FK_ts])

    # Initial state.
    state = odil.State(
        fields={
            'coeff': odil.Array([D_ch, rho_ch,CM_pos[0],CM_pos[1],CM_pos[2],th_down_s,th_up_s, gamma_ch]),
            # Initial trajectory.
            'x': odil.Field(domain.points('x'), loc='cccc'),
            'y': odil.Field(domain.points('y'), loc='cccc'),
            'z': odil.Field(domain.points('z'), loc='cccc'),
            'c': odil.Field(FK_ts, loc='cccc')
        })
    state = domain.init_state(state)
    TS = FK_ts
    return problem, state


def main():
    global problem, args, wm_data, gm_data, csf_data, outside_skull_mask, pet_w

    args = parse_args()
    default_outdir = args.output_dir
    setattr(args, 'outdir', default_outdir)   
    odil.setup_outdir(args,[])
    args.Nt = args.Nx if args.Nt is None else args.Nt
    args.Ny = args.Nx if args.Ny is None else args.Ny
    problem, state = make_problem(args)

    callback = odil.make_callback(problem, args, plot_func=plot, report_func=report_func, history_func=history_func)
    odil.optimize(args, args.optimizer, problem, state, callback)
    
   ######save solution
    domain = problem.domain
    mod = domain.mod
    # Time slices.
    tx, ty, tz = state_to_traj(domain, state, mod=mod)
    tx, ty, tz = np.array(tx), np.array(ty), np.array(tz)

    # Initialize empty lists to store the data for each tissue type
    wm_results = []
    gm_results = []
    csf_results = []

    if args.save_last_timestep_solution:
        print("Computing and saving solution for the given time point...")
        # Compute Euler form of 'c'
        c_lagrange = np.array(transform_c(domain.field(state, 'c'), mod))
        c_euler_last_slice = lagrange_to_euler_single_slice(c_lagrange, tx, ty, tz, domain, -1)

        # Restore final step arrays
        c_euler_restored = restore(c_euler_last_slice)
        
        # **Add this single line to invert the X and Y axes: **
        c_euler_restored = c_euler_restored[::-1, ::-1, :]

        # Save data
        data_dict = {
            'c_euler': c_euler_restored
        }

        # Create a nibabel NIfTI image
        c_euler_nifti = nib.Nifti1Image(c_euler_restored, stored_affine)

        # Save
        nifti_filename = 'c_euler_last_timestep.nii'
        nib.save(c_euler_nifti, nifti_filename)
        print(f"NIfTI file saved as {nifti_filename}")

    else:
        print("Not saving solution as the flag is not set.")



    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
