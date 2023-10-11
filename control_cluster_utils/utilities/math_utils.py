# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of ControlClusterUtils and distributed under the General Public License version 2 license.
# 
# ControlClusterUtils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# ControlClusterUtils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with ControlClusterUtils.  If not, see <http://www.gnu.org/licenses/>.
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
