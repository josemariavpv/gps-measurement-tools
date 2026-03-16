"""Earth-rotation correction for signal flight time (FlightTimeCorrection).

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


def flight_time_correction(x_e, dt_flight_seconds):
    """Compute Earth-rotation–corrected satellite ECEF position.

    Rotates the satellite ECEF coordinates (at time of transmission) by the
    Earth's rotation angle during the signal flight time.

    Reference: IS-GPS-200, 20.3.3.4.3.3.2 Earth-Centred, Inertial Coordinate System.

    Parameters
    ----------
    x_e : array_like, shape (3,)
        Satellite ECEF position at time of transmission (metres).
    dt_flight_seconds : float
        Signal flight time in seconds.

    Returns
    -------
    x_e_rot : ndarray, shape (3,)
        Rotated satellite ECEF position (metres) at time of reception.
    """
    theta = GpsConstants.WE * dt_flight_seconds

    r3 = np.array([
        [ np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0,              0,             1],
    ])
    x_e_rot = r3 @ np.asarray(x_e, dtype=float).flatten()
    return x_e_rot
