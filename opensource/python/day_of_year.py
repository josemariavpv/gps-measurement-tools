"""Day of year computation.

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
from .julian_day import julian_day


def day_of_year(utc_time):
    """Return the day number of the year.

    Parameters
    ----------
    utc_time : array_like, shape (6,)
        [year, month, day, hours, minutes, seconds].

    Returns
    -------
    day_number : float
        Day number within the year (1-based).
    """
    utc_time = np.asarray(utc_time, dtype=float)
    if utc_time.shape != (6,):
        raise ValueError('utc_time must be a 1-D array of length 6 for day_of_year')

    j_day = julian_day([[utc_time[0], utc_time[1], utc_time[2], 0, 0, 0]])[0]
    j_day_jan1 = julian_day([[utc_time[0], 1, 1, 0, 0, 0]])[0]
    return j_day - j_day_jan1 + 1
