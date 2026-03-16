"""LLA to ECEF coordinate conversion (Lla2Xyz).

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


def lla2xyz(lla_deg_deg_m):
    """Transform latitude, longitude, altitude to ECEF coordinates.

    Parameters
    ----------
    lla_deg_deg_m : array_like, shape (m, 3)
        Each row is [lat_deg, lon_deg, alt_m].

    Returns
    -------
    xyz_m : ndarray, shape (m, 3)
        ECEF coordinates in metres.
    """
    lla = np.atleast_2d(np.asarray(lla_deg_deg_m, dtype=float))
    if lla.shape[1] != 3:
        raise ValueError('Input lla_deg_deg_m must have three columns')

    lat_deg = lla[:, 0]
    lon_deg = lla[:, 1]
    alt_m   = lla[:, 2]

    d2r  = np.pi / 180.0
    clat = np.cos(lat_deg * d2r)
    clon = np.cos(lon_deg * d2r)
    slat = np.sin(lat_deg * d2r)
    slon = np.sin(lon_deg * d2r)

    r0 = GpsConstants.EARTHSEMIMAJOR / np.sqrt(
        1.0 - GpsConstants.EARTHECCEN2 * slat * slat)

    x_m = (alt_m + r0) * clat * clon
    y_m = (alt_m + r0) * clat * slon
    z_m = (alt_m + r0 * (1.0 - GpsConstants.EARTHECCEN2)) * slat

    return np.column_stack([x_m, y_m, z_m])
