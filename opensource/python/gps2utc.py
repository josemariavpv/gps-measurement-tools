"""GPS to UTC time conversion.

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
from .leap_seconds import leap_seconds


_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _fct2ymdhms(fct_seconds):
    """Convert GPS full cycle time (seconds) to [year, month, day, hh, mm, ss].

    Utility function for gps2utc.
    """
    HOURSEC = 3600
    MINSEC = 60

    fct_seconds = np.atleast_1d(np.array(fct_seconds, dtype=float))
    m = len(fct_seconds)

    days = np.floor(fct_seconds / GpsConstants.DAYSEC) + 6  # days since 1980/1/1
    years = np.zeros(m) + 1980
    leap = np.ones(m)  # 1980 was a leap year

    while np.any(days > (leap + 365)):
        idx = days > (leap + 365)
        days[idx] -= leap[idx] + 365
        years[idx] += 1
        leap[idx] = (np.remainder(years[idx], 4) == 0).astype(float)

    time_out = np.zeros((m, 6))
    time_out[:, 0] = years

    for i in range(m):
        month_days = _MONTH_DAYS[:]
        if int(np.remainder(years[i], 4)) == 0:
            month_days[1] = 29  # leap year February
        month = 1
        d = days[i]
        while d > month_days[month - 1]:
            d -= month_days[month - 1]
            month += 1
        time_out[i, 1] = month
        time_out[i, 2] = d

    since_midnight = np.remainder(fct_seconds, GpsConstants.DAYSEC)
    time_out[:, 3] = np.fix(since_midnight / HOURSEC)
    last_hour_secs = np.remainder(since_midnight, HOURSEC)
    time_out[:, 4] = np.fix(last_hour_secs / MINSEC)
    time_out[:, 5] = np.remainder(last_hour_secs, MINSEC)

    return time_out


def gps2utc(gps_time=None, fct_seconds=None):
    """Convert GPS time (week & seconds), or Full Cycle Time (seconds) to UTC.

    Parameters
    ----------
    gps_time : array_like, shape (m, 2), optional
        Each row is [gps_week, gps_seconds]. Ignored if fct_seconds is provided.
    fct_seconds : array_like, shape (m,), optional
        Full Cycle Time in seconds since GPS epoch.

    Returns
    -------
    utc_time : ndarray, shape (m, 6)
        Each row is [year, month, day, hours, minutes, seconds].
    """
    if fct_seconds is None:
        if gps_time is None:
            raise ValueError('Either gps_time or fct_seconds must be provided')
        gps_time = np.atleast_2d(np.array(gps_time, dtype=float))
        if gps_time.shape[1] != 2:
            raise ValueError('gps_time must have two columns')
        fct_seconds = gps_time[:, 0] * GpsConstants.WEEKSEC + gps_time[:, 1]
    else:
        fct_seconds = np.atleast_1d(np.array(fct_seconds, dtype=float))

    # fct at 2100/1/1 00:00:00, not counting leap seconds
    fct2100 = np.array([[6260, 432000]], dtype=float)
    fct2100_val = fct2100[0, 0] * GpsConstants.WEEKSEC + fct2100[0, 1]
    if np.any(fct_seconds < 0) or np.any(fct_seconds >= fct2100_val):
        raise ValueError(
            'gps_time must be in this range: [0,0] <= gps_time < [6260, 432000]')

    # Algorithm for handling leap seconds:
    # 1) convert gps_time to time (with no leap seconds)
    time = _fct2ymdhms(fct_seconds)
    # 2) look up leap seconds
    ls = leap_seconds(time)
    # 3) convert gps_time - ls to time_mls
    time_mls = _fct2ymdhms(fct_seconds - ls)
    # 4) look up leap seconds for time_mls
    ls1 = leap_seconds(time_mls)
    # 5) if ls1 != ls, convert (gps_time - ls1) to UTC Time
    if np.all(ls1 == ls):
        utc_time = time_mls
    else:
        utc_time = _fct2ymdhms(fct_seconds - ls1)

    return utc_time
