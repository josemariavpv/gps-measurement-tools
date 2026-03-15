"""Leap seconds lookup.

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


# UTC table: each row is the UTC time right after a new leap second occurred.
# LATEST LEAP SECOND IN THE TABLE = 31 Dec 2016. On 1 Jan 2017: GPS-UTC=18s
# No further leap seconds have been inserted between 2017 and 2026.
# (The CGPM voted in November 2022 to eliminate leap seconds by 2035.)
# See IERS Bulletin C, https://hpiers.obspm.fr/iers/bul/bulc/bulletinc.dat
_UTC_TABLE = np.array([
    [1982, 1, 1, 0, 0, 0],
    [1982, 7, 1, 0, 0, 0],
    [1983, 7, 1, 0, 0, 0],
    [1985, 7, 1, 0, 0, 0],
    [1988, 1, 1, 0, 0, 0],
    [1990, 1, 1, 0, 0, 0],
    [1991, 1, 1, 0, 0, 0],
    [1992, 7, 1, 0, 0, 0],
    [1993, 7, 1, 0, 0, 0],
    [1994, 7, 1, 0, 0, 0],
    [1996, 1, 1, 0, 0, 0],
    [1997, 7, 1, 0, 0, 0],
    [1999, 1, 1, 0, 0, 0],
    [2006, 1, 1, 0, 0, 0],
    [2009, 1, 1, 0, 0, 0],
    [2012, 7, 1, 0, 0, 0],
    [2015, 7, 1, 0, 0, 0],
    [2017, 1, 1, 0, 0, 0],
], dtype=float)


def leap_seconds(utc_time):
    """Find the number of leap seconds since the GPS Epoch.

    Parameters
    ----------
    utc_time : array_like, shape (m, 6)
        Each row is [year, month, day, hours, minutes, seconds].
        year valid range: 1980 <= year <= 2099.

    Returns
    -------
    leap_secs : ndarray, shape (m,)
        Number of leap seconds between the GPS Epoch and each utc_time row.
    """
    utc_time = np.atleast_2d(np.array(utc_time, dtype=float))
    m_rows = utc_time.shape[0]
    if utc_time.shape[1] != 6:
        raise ValueError('utc_time input must have 6 columns')

    table_j_days = julian_day(_UTC_TABLE) - GpsConstants.GPSEPOCHJD
    table_seconds = table_j_days * GpsConstants.DAYSEC

    j_days = julian_day(utc_time) - GpsConstants.GPSEPOCHJD
    time_seconds = j_days * GpsConstants.DAYSEC

    leap_secs = np.zeros(m_rows)
    for i in range(m_rows):
        leap_secs[i] = np.sum(table_seconds <= time_seconds[i])

    return leap_secs
