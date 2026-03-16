"""ECEF to LLA coordinate conversion (Xyz2Lla).

Author: Frank van Diggelen
Open Source code for processing Android GNSS Measurements
"""

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from .gps_constants import GpsConstants


def xyz2lla(xyz_m):
    """Transform ECEF coordinates to latitude, longitude, altitude.

    Algorithm: Hoffman-Wellenhof, Lichtenegger & Collins "GPS Theory & Practice".

    Parameters
    ----------
    xyz_m : array_like, shape (m, 3)
        ECEF coordinates in metres.

    Returns
    -------
    lla_deg_deg_m : ndarray, shape (m, 3)
        Each row is [lat_deg, lon_deg, alt_m].
    """
    xyz = np.atleast_2d(np.asarray(xyz_m, dtype=float)).copy()
    if xyz.shape[1] != 3:
        raise ValueError('Input xyz_m must have three columns')

    r2d = 180.0 / np.pi

    # if x and y are both zero, result is undefined → set to NaN
    i_zero = (xyz[:, 0] == 0) & (xyz[:, 1] == 0)
    xyz[i_zero, :] = np.nan

    x_m = xyz[:, 0]
    y_m = xyz[:, 1]
    z_m = xyz[:, 2]

    a  = GpsConstants.EARTHSEMIMAJOR
    a2 = a ** 2
    e2 = GpsConstants.EARTHECCEN2
    b2 = a2 * (1.0 - e2)
    b  = np.sqrt(b2)
    ep2 = (a2 - b2) / b2
    p   = np.sqrt(x_m ** 2 + y_m ** 2)

    s1 = z_m * a
    s2 = p * b
    h  = np.sqrt(s1 ** 2 + s2 ** 2)
    sin_theta = s1 / h
    cos_theta = s2 / h

    s1  = z_m  + ep2 * b * sin_theta ** 3
    s2  = p    - a  * e2 * cos_theta ** 3
    h   = np.sqrt(s1 ** 2 + s2 ** 2)
    sin_lat = s1 / h
    cos_lat = s2 / h

    lat_deg = np.arctan2(sin_lat, cos_lat) * r2d

    n_val = a2 / np.sqrt(a2 * cos_lat ** 2 + b2 * sin_lat ** 2)
    alt_m = p / cos_lat - n_val

    lon_deg = np.arctan2(y_m, x_m)
    lon_deg = np.remainder(lon_deg, 2.0 * np.pi) * r2d
    lon_deg[lon_deg > 180.0] -= 360.0

    return np.column_stack([lat_deg, lon_deg, alt_m])
