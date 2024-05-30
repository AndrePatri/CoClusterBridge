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
import torch
from typing import List

def incremental_rotate(q_initial: torch.Tensor, 
                    d_angle: float, 
                    axis: List) -> torch.Tensor:
    """
    Incrementally rotate a quaternion `q_initial` by `d_angle` radians about `axis`.
    Parameters:
    - q_initial (torch.Tensor): Initial quaternion.
    - d_angle (float): Angle by which to rotate, in radians.
    - axis (torch.Tensor): Axis about which to rotate.
    Returns:
    - torch.Tensor: Resulting quaternion after rotation.
    """
    d_angle_tensor = torch.tensor(d_angle)
    # Compute the quaternion representation of the incremental rotation
    q_incremental = torch.tensor([torch.cos(d_angle_tensor / 2),
                            axis[0] * torch.sin(d_angle_tensor / 2),
                            axis[1] * torch.sin(d_angle_tensor / 2),
                            axis[2] * torch.sin(d_angle_tensor / 2)])
    
    # Normalize the quaternion
    q_incremental /= torch.linalg.norm(q_incremental)
    
    # Compute the final orientation of the base by multiplying the quaternions
    q_result = quaternion_multiply(q_incremental, q_initial)
    
    return q_result

def quaternion_multiply(q1: torch.Tensor, 
                q2: torch.Tensor):
    
    """
    Multiply two quaternions q1 and q2.
    Assumes quaternions are represented as 1D tensors: [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.tensor([w, x, y, z])

def w2hor_frame(v_w: torch.Tensor,
        q_b: torch.Tensor,
        v_out: torch.Tensor):
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
    
    norm = torch.sqrt(R_11 ** 2 + R_21 ** 2)
    x_proj_x = R_11 / norm
    x_proj_y = R_21 / norm
    
    y_proj_x = -x_proj_y
    y_proj_y = x_proj_x
        
    v_out[:, 0] = v_w[:, 0] * x_proj_x + v_w[:, 1] * x_proj_y
    v_out[:, 1] = v_w[:, 0] * y_proj_x + v_w[:, 1] * y_proj_y
    v_out[:, 2] = v_w[:, 2]  # z-component remains the same

def base2world_frame(v_b: torch.Tensor, q_b: torch.Tensor, v_out: torch.Tensor):
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

def base2world_frame_new(v_b: torch.Tensor, q_b: torch.Tensor, v_out: torch.Tensor):
    """
    Transforms a velocity vector expressed in the base frame to
    the WORLD frame using the given quaternion that describes the orientation
    of the base with respect to the world frame. The result is written in v_out.
    """
    # Normalize quaternion
    # q_b_norm = q_b / q_b.norm(dim=1, keepdim=True)
    
    # Extract quaternion components
    q_w, q_i, q_j, q_k = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
    
    # Extract velocity components in the base frame
    v_x, v_y, v_z = v_b[:, 0], v_b[:, 1], v_b[:, 2]
    
    # Calculate quaternion conjugate
    q_conj = torch.stack([q_w, -q_i, -q_j, -q_k], dim=1)
    
    # Quaternion multiplication to rotate velocity vector
    v_out[:, 0] = v_x * (q_w ** 2 + q_i ** 2 - q_j ** 2 - q_k ** 2) + \
                  2 * (v_y * (q_i * q_j - q_w * q_k) + v_z * (q_i * q_k + q_w * q_j))
                  
    v_out[:, 1] = 2 * (v_x * (q_i * q_j + q_w * q_k) + \
                       v_y * (q_w ** 2 - q_i ** 2 + q_j ** 2 - q_k ** 2)) + \
                  v_z * (2 * (q_j * q_k - q_i * q_w))
                  
    v_out[:, 2] = 2 * (v_x * (q_i * q_k - q_w * q_j) + \
                       v_y * (q_j * q_k + q_i * q_w)) + \
                  v_z * (q_w ** 2 - q_i ** 2 - q_j ** 2 + q_k ** 2)
    
if __name__ == "__main__":  

    n_envs = 5
    v_b = torch.randn(n_envs, 3)
    q_b = torch.randn(n_envs, 4)
    q_b_norm = q_b / q_b.norm(dim=1, keepdim=True)
    v_out_old = torch.empty_like(v_b)
    v_out_new = torch.empty_like(v_b)

    # Use both implementations
    base2world_frame_old(v_b, q_b_norm, v_out_old)
    base2world_frame_new(v_b, q_b_norm, v_out_new)

    # Print the results
    print("Old implementation result:", v_out_old)
    print("New implementation result:", v_out_new)

    # Compute the difference between the results
    difference = torch.abs(v_out_old - v_out_new)
    print("Difference:", difference)
    print("Max difference:", torch.max(difference))
