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

