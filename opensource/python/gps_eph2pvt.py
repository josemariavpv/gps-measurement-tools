"""GPS satellite PVT (position, velocity, clock) from ephemeris (GpsEph2Pvt).

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
from .gps_eph2xyz import gps_eph2xyz


def gps_eph2pvt(gps_eph, gps_time):
    """Calculate satellite ECEF position, velocity and clock bias/rate.

    Velocity and clock rate are computed by finite differencing:
        v = (x(t+0.5) - x(t-0.5)) / 1  second

    Parameters
    ----------
    gps_eph : list of dict
        Vector of GPS ephemeris dicts as returned by read_rinex_nav().
    gps_time : array_like, shape (n, 2)
        GPS time [gps_week, ttx_sec] at time of transmission for each satellite.
        n must equal len(gps_eph).

    Returns
    -------
    xyz_m : ndarray, shape (n, 3)
        ECEF position of each satellite (metres).
    dtsv_s : ndarray, shape (n,)
        Satellite clock error (seconds).
    v_mps : ndarray, shape (n, 3)
        ECEF velocity (m/s).
    dtsv_dot : ndarray, shape (n,)
        Satellite clock error rate (s/s).
    """
    gps_time = np.atleast_2d(np.asarray(gps_time, dtype=float))

    xyz_m, dtsv_s = gps_eph2xyz(gps_eph, gps_time)
    if xyz_m is None or len(xyz_m) == 0:
        return None, None, None, None

    t_plus  = np.column_stack([gps_time[:, 0], gps_time[:, 1] + 0.5])
    t_minus = np.column_stack([gps_time[:, 0], gps_time[:, 1] - 0.5])

    xyz_plus,  dtsv_plus  = gps_eph2xyz(gps_eph, t_plus)
    xyz_minus, dtsv_minus = gps_eph2xyz(gps_eph, t_minus)

    v_mps    = xyz_plus  - xyz_minus   # 1-second difference → m/s
    dtsv_dot = dtsv_plus - dtsv_minus

    return xyz_m, dtsv_s, v_mps, dtsv_dot
