"""GPS satellite clock bias from ephemeris (GpsEph2Dtsv).

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
from .kepler import kepler


def gps_eph2dtsv(gps_eph, t_s):
    """Calculate satellite clock bias from GPS ephemeris.

    Parameters
    ----------
    gps_eph : list of dict
        Vector of GPS ephemeris dicts as returned by read_rinex_nav().
    t_s : array_like
        GPS time of week (seconds) at time of transmission.
        If gps_eph has more than one element, t_s must be the same length.

    Returns
    -------
    dtsv_s : ndarray
        Satellite clock bias in seconds. GPS time = satellite time - dtsv_s.
    """
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float)).flatten()
    p = len(gps_eph)
    pt = len(t_s)
    if p > 1 and pt != p:
        raise ValueError(
            'If gps_eph is a vector, t_s must be the same length')

    tgd       = np.array([e['TGD']     for e in gps_eph], dtype=float)
    toc       = np.array([e['Toc']     for e in gps_eph], dtype=float)
    af2       = np.array([e['af2']     for e in gps_eph], dtype=float)
    af1       = np.array([e['af1']     for e in gps_eph], dtype=float)
    af0       = np.array([e['af0']     for e in gps_eph], dtype=float)
    delta_n   = np.array([e['Delta_n'] for e in gps_eph], dtype=float)
    m0        = np.array([e['M0']      for e in gps_eph], dtype=float)
    e_ecc     = np.array([e['e']       for e in gps_eph], dtype=float)
    asqrt     = np.array([e['Asqrt']   for e in gps_eph], dtype=float)
    toe       = np.array([e['Toe']     for e in gps_eph], dtype=float)

    tk = t_s - toe
    tk[tk > 302400.0]  -= GpsConstants.WEEKSEC
    tk[tk < -302400.0] += GpsConstants.WEEKSEC

    a  = asqrt ** 2
    n0 = np.sqrt(GpsConstants.mu / a ** 3)
    n  = n0 + delta_n
    mk_ = m0 + n * tk
    ek  = kepler(mk_, e_ecc)

    dt = t_s - toc
    dt[dt > 302400.0]  -= GpsConstants.WEEKSEC
    dt[dt < -302400.0] += GpsConstants.WEEKSEC

    dtsv_s = (af0 + af1 * dt + af2 * dt ** 2
              + GpsConstants.FREL * e_ecc * asqrt * np.sin(ek)
              - tgd)
    return dtsv_s
