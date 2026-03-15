"""Find closest GPS ephemeris for a set of satellite IDs (ClosestGpsEph).

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


def closest_gps_eph(all_gps_eph, sv_ids, fct_seconds):
    """Find the best matching GPS ephemeris for each satellite ID.

    For each satellite in sv_ids, find the ephemeris record in all_gps_eph
    whose Toe is closest to fct_seconds and within the fit interval.

    Parameters
    ----------
    all_gps_eph : list of dict
        All GPS ephemeris records.
    sv_ids : array_like
        Satellite IDs (PRN numbers) to search for.
    fct_seconds : float
        Full-cycle GPS time (seconds) of the desired epoch.

    Returns
    -------
    gps_eph : list of dict
        Selected ephemeris records (one per matched satellite).
    i_sv : list of int
        Indices into sv_ids corresponding to each returned ephemeris.
    """
    sv_ids = np.atleast_1d(np.asarray(sv_ids, dtype=int))
    all_prn = np.array([e['PRN'] for e in all_gps_eph], dtype=int)

    gps_eph = []
    i_sv = []

    for i, prn in enumerate(sv_ids):
        mask = all_prn == prn
        if not np.any(mask):
            continue
        eph_sv = [all_gps_eph[k] for k in np.where(mask)[0]]

        fit_hours = np.array([e['Fit_interval'] for e in eph_sv], dtype=float)
        fit_hours[fit_hours == 0] = 4.0

        fct_toe = np.array(
            [e['GPS_Week'] * GpsConstants.WEEKSEC + e['Toe'] for e in eph_sv],
            dtype=float)
        age = np.abs(fct_toe - fct_seconds)
        i_min = int(np.argmin(age))
        if age[i_min] < (fit_hours[i_min] / 2.0) * 3600.0:
            gps_eph.append(eph_sv[i_min])
            i_sv.append(i)
        else:
            age_hours = (fct_toe[i_min] - fct_seconds) / 3600.0
            print(f'No valid ephemeris found for svId {prn}, '
                  f'closest Toe is {age_hours:.1f} hours away.')

    return gps_eph, i_sv
