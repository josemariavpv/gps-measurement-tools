"""GPS satellite ECEF position from ephemeris (GpsEph2Xyz).

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


def gps_eph2xyz(gps_eph, gps_time):
    """Calculate satellite ECEF coordinates from GPS ephemeris.

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
        ECEF coordinates of each satellite (metres).
    dtsv_s : ndarray, shape (n,)
        Satellite clock error (seconds).
    """
    gps_time = np.atleast_2d(np.asarray(gps_time, dtype=float))
    p = len(gps_eph)
    if gps_time.shape[0] != p:
        raise ValueError('gps_time must have one row per ephemeris')

    gps_week = gps_time[:, 0]
    ttx_sec  = gps_time[:, 1]

    fit_hours = np.array([e['Fit_interval'] for e in gps_eph], dtype=float)
    fit_hours[fit_hours == 0] = 2.0
    fit_seconds = fit_hours * 3600.0

    tgd       = np.array([e['TGD']       for e in gps_eph], dtype=float)
    toc       = np.array([e['Toc']       for e in gps_eph], dtype=float)
    af2       = np.array([e['af2']       for e in gps_eph], dtype=float)
    af1       = np.array([e['af1']       for e in gps_eph], dtype=float)
    af0       = np.array([e['af0']       for e in gps_eph], dtype=float)
    crs       = np.array([e['Crs']       for e in gps_eph], dtype=float)
    delta_n   = np.array([e['Delta_n']   for e in gps_eph], dtype=float)
    m0        = np.array([e['M0']        for e in gps_eph], dtype=float)
    cuc       = np.array([e['Cuc']       for e in gps_eph], dtype=float)
    e_ecc     = np.array([e['e']         for e in gps_eph], dtype=float)
    cus       = np.array([e['Cus']       for e in gps_eph], dtype=float)
    asqrt     = np.array([e['Asqrt']     for e in gps_eph], dtype=float)
    toe       = np.array([e['Toe']       for e in gps_eph], dtype=float)
    cic       = np.array([e['Cic']       for e in gps_eph], dtype=float)
    omega_big = np.array([e['OMEGA']     for e in gps_eph], dtype=float)
    cis       = np.array([e['Cis']       for e in gps_eph], dtype=float)
    i0        = np.array([e['i0']        for e in gps_eph], dtype=float)
    crc       = np.array([e['Crc']       for e in gps_eph], dtype=float)
    omega     = np.array([e['omega']     for e in gps_eph], dtype=float)
    omega_dot = np.array([e['OMEGA_DOT'] for e in gps_eph], dtype=float)
    idot      = np.array([e['IDOT']      for e in gps_eph], dtype=float)
    eph_week  = np.array([e['GPS_Week']  for e in gps_eph], dtype=float)

    # Time since time of applicability (account for week roll-over)
    tk = (gps_week - eph_week) * GpsConstants.WEEKSEC + (ttx_sec - toe)

    outside = np.abs(tk) > fit_seconds
    if np.any(outside):
        print(f'WARNING in gps_eph2xyz: {np.sum(outside)} times outside fit interval.')

    a  = asqrt ** 2
    n0 = np.sqrt(GpsConstants.mu / a ** 3)
    n  = n0 + delta_n
    mk_ = m0 + n * tk
    ek  = kepler(mk_, e_ecc)

    # Clock bias
    dt = (gps_week - eph_week) * GpsConstants.WEEKSEC + (ttx_sec - toc)
    sin_ek = np.sin(ek)
    cos_ek = np.cos(ek)
    dtsv_s = (af0 + af1 * dt + af2 * dt ** 2
              + GpsConstants.FREL * e_ecc * asqrt * sin_ek
              - tgd)

    # True anomaly
    vk = np.arctan2(np.sqrt(1.0 - e_ecc ** 2) * sin_ek / (1.0 - e_ecc * cos_ek),
                    (cos_ek - e_ecc) / (1.0 - e_ecc * cos_ek))
    phi_k = vk + omega

    sin_2phi = np.sin(2.0 * phi_k)
    cos_2phi = np.cos(2.0 * phi_k)

    duk = cus * sin_2phi + cuc * cos_2phi
    drk = crc * cos_2phi + crs * sin_2phi
    dik = cic * cos_2phi + cis * sin_2phi

    uk = phi_k + duk
    rk = a * ((1.0 - e_ecc ** 2) / (1.0 + e_ecc * np.cos(vk))) + drk
    ik = i0 + idot * tk + dik

    sin_uk = np.sin(uk)
    cos_uk = np.cos(uk)
    xkp = rk * cos_uk
    ykp = rk * sin_uk

    # Corrected longitude of ascending node
    wk = omega_big + (omega_dot - GpsConstants.WE) * tk - GpsConstants.WE * toe

    sin_wk = np.sin(wk)
    cos_wk = np.cos(wk)
    sin_ik = np.sin(ik)
    cos_ik = np.cos(ik)

    xyz_m = np.zeros((p, 3))
    xyz_m[:, 0] = xkp * cos_wk - ykp * cos_ik * sin_wk
    xyz_m[:, 1] = xkp * sin_wk + ykp * cos_ik * cos_wk
    xyz_m[:, 2] = ykp * sin_ik

    return xyz_m, dtsv_s
