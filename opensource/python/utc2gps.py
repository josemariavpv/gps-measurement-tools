"""UTC to GPS time conversion.

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
from .julian_day import julian_day
from .leap_seconds import leap_seconds


def _check_utc_time_inputs(utc_time):
    """Validate UTC time inputs."""
    if utc_time.shape[1] != 6:
        raise ValueError('utc_time must have 6 columns')
    x = utc_time[:, :3]
    if np.any((x - np.fix(x)) != 0):
        raise ValueError('year, month & day must be integers')
    if np.any(utc_time[:, 0] < 1980) or np.any(utc_time[:, 0] > 2099):
        raise ValueError('year must have values in the range: [1980:2099]')
    if np.any(utc_time[:, 1] < 1) or np.any(utc_time[:, 1] > 12):
        raise ValueError('The month in utc_time must be a number in the set [1:12]')
    if np.any(utc_time[:, 2] < 1) or np.any(utc_time[:, 2] > 31):
        raise ValueError('The day in utc_time must be a number in the set [1:31]')
    if np.any(utc_time[:, 3] < 0) or np.any(utc_time[:, 3] >= 24):
        raise ValueError('The hour in utc_time must be in the range [0, 24)')
    if np.any(utc_time[:, 4] < 0) or np.any(utc_time[:, 4] >= 60):
        raise ValueError('The minutes in utc_time must be in the range [0, 60)')
    if np.any(utc_time[:, 5] < 0) or np.any(utc_time[:, 5] > 60):
        raise ValueError('The seconds in utc_time must be in the range [0, 60]')


def utc2gps(utc_time):
    """Convert UTC date and time to GPS week and seconds.

    Parameters
    ----------
    utc_time : array_like, shape (m, 6)
        Each row is [year, month, day, hours, minutes, seconds].
        year must be four digits; valid range: 1980 <= year <= 2099.

    Returns
    -------
    gps_time : ndarray, shape (m, 2)
        Each row is [gps_week, gps_seconds].
    fct_seconds : ndarray, shape (m,)
        Full cycle time = seconds since GPS epoch (1980-01-06 00:00 UTC).
    """
    HOURSEC = 3600
    MINSEC = 60

    utc_time = np.atleast_2d(np.array(utc_time, dtype=float))
    _check_utc_time_inputs(utc_time)

    days_since_epoch = np.floor(julian_day(utc_time) - GpsConstants.GPSEPOCHJD)
    gps_week = np.fix(days_since_epoch / 7.0)
    day_of_week = np.remainder(days_since_epoch, 7)

    gps_seconds = (day_of_week * GpsConstants.DAYSEC
                   + utc_time[:, 3] * HOURSEC
                   + utc_time[:, 4] * MINSEC
                   + utc_time[:, 5])

    gps_week = gps_week + np.fix(gps_seconds / GpsConstants.WEEKSEC)
    gps_seconds = np.remainder(gps_seconds, GpsConstants.WEEKSEC)

    # add leap seconds
    leap_secs = leap_seconds(utc_time)
    fct_seconds = gps_week * GpsConstants.WEEKSEC + gps_seconds + leap_secs

    gps_week = np.fix(fct_seconds / GpsConstants.WEEKSEC)
    iz = gps_week == 0
    gps_seconds_out = np.zeros_like(fct_seconds)
    gps_seconds_out[iz] = fct_seconds[iz]
    niz = ~iz
    gps_seconds_out[niz] = np.remainder(
        fct_seconds[niz], gps_week[niz] * GpsConstants.WEEKSEC)

    gps_time = np.column_stack([gps_week, gps_seconds_out])

    # assert consistency
    assert np.all(fct_seconds == gps_week * GpsConstants.WEEKSEC + gps_seconds_out), \
        'Error in computing gps_week, gps_seconds'

    return gps_time, fct_seconds
