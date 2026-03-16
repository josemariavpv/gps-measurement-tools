"""ECEF to NED rotation matrix (RotEcef2Ned).

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


def rot_ecef2ned(lat_deg, lon_deg):
    """Return the 3×3 rotation matrix from ECEF to NED.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geodetic latitude and longitude in degrees.

    Returns
    -------
    re2n : ndarray, shape (3, 3)
        Rotation matrix.  ``v_ned = re2n @ v_ecef``.
    """
    d2r = np.pi / 180.0
    lat = float(lat_deg) * d2r
    lon = float(lon_deg) * d2r

    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)

    re2n = np.array([
        [-slat * clon, -slat * slon,  clat],
        [-slon,         clon,         0.0 ],
        [-clat * clon, -clat * slon, -slat],
    ])
    return re2n
