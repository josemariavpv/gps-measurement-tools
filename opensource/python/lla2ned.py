"""LLA to NED difference vector (Lla2Ned).

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
from .lla2xyz import lla2xyz
from .xyz2lla import xyz2lla
from .rot_ecef2ned import rot_ecef2ned


def lla2ned(lla1_deg_deg_m, lla2_deg_deg_m):
    """Compute lla1 − lla2 in North-East-Down (NED) coordinates.

    Parameters
    ----------
    lla1_deg_deg_m : array_like, shape (m, 3)
        Query positions [lat_deg, lon_deg, alt_m].
    lla2_deg_deg_m : array_like, shape (m, 3) or (1, 3)
        Reference position(s).  If one row, it is broadcast to all rows of lla1.

    Returns
    -------
    ned_m : ndarray, shape (m, 3)
        [north, east, down] differences in metres.
    """
    lla1 = np.atleast_2d(np.asarray(lla1_deg_deg_m, dtype=float))
    lla2 = np.atleast_2d(np.asarray(lla2_deg_deg_m, dtype=float))

    m1 = lla1.shape[0]
    if lla2.shape[0] == 1:
        lla2 = np.tile(lla2, (m1, 1))
    elif lla2.shape[0] != m1:
        raise ValueError(
            'Second input must have one row or the same number of rows as first input')

    if lla1.shape[1] != 3 or lla2.shape[1] != 3:
        raise ValueError('Both input matrices must have 3 columns')

    xyz1 = lla2xyz(lla1)
    xyz2 = lla2xyz(lla2)
    ref_xyz = (xyz1 + xyz2) / 2.0
    ref_lla = xyz2lla(ref_xyz)

    north = np.zeros(m1)
    east  = np.zeros(m1)
    for i in range(m1):
        ce2n = rot_ecef2ned(ref_lla[i, 0], ref_lla[i, 1])
        v = ce2n @ (xyz1[i] - xyz2[i])
        north[i] = v[0]
        east[i]  = v[1]

    down = -lla1[:, 2] + lla2[:, 2]
    return np.column_stack([north, east, down])
