# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of CoClusterBridge and distributed under the General Public License version 2 license.
# 
# CoClusterBridge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# CoClusterBridge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CoClusterBridge.  If not, see <http://www.gnu.org/licenses/>.
# 
import numpy as np

def incremental_rotate(q_initial, d_angle, axis):
    """
    Incrementally rotate a quaternion `q_initial` by `d_angle` radians about `axis`.
    Parameters:
    - q_initial (ndarray): Initial quaternion.
    - d_angle (float): Angle by which to rotate, in radians.
    - axis (ndarray): Axis about which to rotate.
    Returns:
    - ndarray: Resulting quaternion after rotation.
    """
    d_angle_tensor = np.array(d_angle)
    # Compute the quaternion representation of the incremental rotation
    q_incremental = np.array([np.cos(d_angle_tensor / 2),
                              axis[0] * np.sin(d_angle_tensor / 2),
                              axis[1] * np.sin(d_angle_tensor / 2),
                              axis[2] * np.sin(d_angle_tensor / 2)])
    
    # Normalize the quaternion
    q_incremental /= np.linalg.norm(q_incremental)
    
    # Compute the final orientation of the base by multiplying the quaternions
    q_result = quaternion_multiply(q_incremental, q_initial)
    
    return q_result

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2.
    Assumes quaternions are represented as 1D arrays: [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return np.array([w, x, y, z])

def w2hor_frame(v_w: np.ndarray,
        q_b: np.ndarray,
        v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in WORLD frame to
    an "horizontal" frame (z aligned as world, x aligned as the projection
    of the x-axis of the base frame described by q_b). This is useful for specifying locomotion
    references in a "game"-like fashion.
    v_out will hold the result
    """

    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
    
    R_11 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_21 = 2 * (q_i * q_j + q_k * q_w)
    
    norm = np.sqrt(R_11 ** 2 + R_21 ** 2)
    x_proj_x = R_11 / norm
    x_proj_y = R_21 / norm
    
    y_proj_x = -x_proj_y
    y_proj_y = x_proj_x
        
    v_out[:, 0] = v_w[:, 0] * x_proj_x + v_w[:, 1] * x_proj_y
    v_out[:, 1] = v_w[:, 0] * y_proj_x + v_w[:, 1] * y_proj_y
    v_out[:, 2] = v_w[:, 2]  # z-component remains the same

def hor2w_frame(v_h: np.ndarray,
        q_b: np.ndarray,
        v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in "horizontal" frame to WORLD
    v_out will hold the result
    """

    # Extract quaternion components
    q_w, q_i, q_j, q_k = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
    
    # Compute rotation matrix elements
    R_11 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_21 = 2 * (q_i * q_j + q_k * q_w)
    
    # Normalize to get projection components
    norm = np.sqrt(R_11 ** 2 + R_21 ** 2)
    x_proj_x = R_11 / norm
    x_proj_y = R_21 / norm
    
    # Orthogonal vector components
    y_proj_x = -x_proj_y
    y_proj_y = x_proj_x
    
    # Transform velocity vector components from horizontal to world frame
    v_out[:, 0] = v_h[:, 0] * x_proj_x + v_h[:, 1] * y_proj_x
    v_out[:, 1] = v_h[:, 0] * x_proj_y + v_h[:, 1] * y_proj_y
    v_out[:, 2] = v_h[:, 2]  # z-component remains the same

def base2world_frame(v_b: np.ndarray, q_b: np.ndarray, v_out: np.ndarray):
    """
    Transforms a velocity vector expressed in the base frame to
    the WORLD frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    q_w, q_i, q_j, q_k = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
    
    R_00 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_01 = 2 * (q_i * q_j - q_k * q_w)
    R_02 = 2 * (q_i * q_k + q_j * q_w)
    
    R_10 = 2 * (q_i * q_j + q_k * q_w)
    R_11 = 1 - 2 * (q_i ** 2 + q_k ** 2)
    R_12 = 2 * (q_j * q_k - q_i * q_w)
    
    R_20 = 2 * (q_i * q_k - q_j * q_w)
    R_21 = 2 * (q_j * q_k + q_i * q_w)
    R_22 = 1 - 2 * (q_i ** 2 + q_j ** 2)
    
    # Extract the velocity components in the base frame
    v_x, v_y, v_z = v_b[:, 0], v_b[:, 1], v_b[:, 2]
    
    # Transform the velocity to the world frame
    v_out[:, 0] = v_x * R_00 + v_y * R_01 + v_z * R_02
    v_out[:, 1] = v_x * R_10 + v_y * R_11 + v_z * R_12
    v_out[:, 2] = v_x * R_20 + v_y * R_21 + v_z * R_22

def world2base_frame(v_w: np.ndarray, q_b: np.ndarray, v_out: np.ndarray,
        is_q_wijk: bool = True):
    """
    Transforms a velocity vector expressed in the WORLD frame to
    the base frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # q_b = q_b / q_b.norm(dim=1, keepdim=True)
    if is_q_wijk:
        q_w, q_i, q_j, q_k = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
    else:
        q_w, q_i, q_j, q_k = q_b[:, 3], q_b[:, 0], q_b[:, 1], q_b[:, 2]

    R_00 = 1 - 2 * (q_j ** 2 + q_k ** 2)
    R_01 = 2 * (q_i * q_j - q_k * q_w)
    R_02 = 2 * (q_i * q_k + q_j * q_w)
    
    R_10 = 2 * (q_i * q_j + q_k * q_w)
    R_11 = 1 - 2 * (q_i ** 2 + q_k ** 2)
    R_12 = 2 * (q_j * q_k - q_i * q_w)
    
    R_20 = 2 * (q_i * q_k - q_j * q_w)
    R_21 = 2 * (q_j * q_k + q_i * q_w)
    R_22 = 1 - 2 * (q_i ** 2 + q_j ** 2)
    
    # Extract the velocity components in the world frame
    v_x, v_y, v_z = v_w[:, 0], v_w[:, 1], v_w[:, 2]
    
    # Transform the velocity to the base frame using the transpose of the rotation matrix
    v_out[:, 0] = v_x * R_00 + v_y * R_10 + v_z * R_20
    v_out[:, 1] = v_x * R_01 + v_y * R_11 + v_z * R_21
    v_out[:, 2] = v_x * R_02 + v_y * R_12 + v_z * R_22

if __name__ == "__main__":  

    n_envs = 5000
    v_b = np.random.rand(n_envs, 3)

    q_b = np.random.rand(n_envs, 4)
    q_b_norm = q_b / np.linalg.norm(q_b, axis=1, keepdims=True)

    v_w = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_b_recovered = np.zeros_like(v_b)  # To hold recovered world frame velocities
    base2world_frame(v_b, q_b_norm, v_w)
    world2base_frame(v_w, q_b_norm, v_b_recovered)
    assert np.allclose(v_b, v_b_recovered, atol=1e-6), "Test failed: v_w_recovered does not match v_b"
    print("Forward test passed: v_b_recovered matches v_b")
    
    v_b2 = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_w_recovered = np.zeros_like(v_b)
    world2base_frame(v_b, q_b_norm, v_b2)
    base2world_frame(v_b2, q_b_norm, v_w_recovered)
    assert np.allclose(v_b, v_w_recovered, atol=1e-6), "Test failed: v_w_recovered does not match v_b"
    print("Backward test passed: v_w_recovered matches v_w")
    
    # test transf. world-horizontal frame
    v_h = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_recovered = np.zeros_like(v_b)
    w2hor_frame(v_b, q_b_norm, v_h)
    hor2w_frame(v_h, q_b_norm, v_recovered)
    assert np.allclose(v_b, v_recovered, atol=1e-6), "Test failed: v_recovered does not match v_b"
    print("horizontal forward frame test passed:  matches ")

    v_w = np.zeros_like(v_b)  # To hold horizontal frame velocities
    v_h_recovered = np.zeros_like(v_b)
    hor2w_frame(v_b, q_b_norm, v_w)
    w2hor_frame(v_w, q_b_norm, v_h_recovered)
    assert np.allclose(v_b, v_h_recovered, atol=1e-6), "Test failed: v_h_recovered does not match v_b"
    print("horizontal backward frame test passed:  matches ")