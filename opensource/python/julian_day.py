"""Julian Day computation.

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


def julian_day(utc_time):
    """Compute Julian Days from UTC time.

    Algorithm from Meeus (1991) Astronomical Algorithms.
    Valid range: 1900 < year < 2100.

    Parameters
    ----------
    utc_time : array_like, shape (m, 6)
        Each row is [year, month, day, hours, minutes, seconds].
        year must be in range (1900, 2100).

    Returns
    -------
    j_days : ndarray, shape (m,)
        Total days in Julian Days (real number of days).
    """
    utc_time = np.atleast_2d(np.array(utc_time, dtype=float))
    if utc_time.shape[1] != 6:
        raise ValueError('utc_time must have 6 columns')

    y = utc_time[:, 0]
    m = utc_time[:, 1]
    d = utc_time[:, 2]
    h = utc_time[:, 3] + utc_time[:, 4] / 60.0 + utc_time[:, 5] / 3600.0

    if np.any(y < 1901) or np.any(y > 2099):
        raise ValueError('utc_time[:, 0] not in allowed range: 1900 < year < 2100')

    y = y.copy()
    m = m.copy()
    i2 = m <= 2
    m[i2] += 12
    y[i2] -= 1

    j_days = (np.floor(365.25 * y)
              + np.floor(30.6001 * (m + 1))
              - 15 + 1720996.5 + d + h / 24.0)
    return j_days
