"""GPS ADR single-difference residuals (GpsAdrResiduals).

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
from .lla2xyz import lla2xyz
from .closest_gps_eph import closest_gps_eph
from .gps_eph2dtsv import gps_eph2dtsv
from .gps_eph2xyz import gps_eph2xyz
from .flight_time_correction import flight_time_correction


def gps_adr_residuals(gnss_meas, all_gps_eph, lla_deg_deg_m=None):
    """Compute single-difference ADR residuals.

    Parameters
    ----------
    gnss_meas : dict
        Processed GNSS measurements from process_gnss_meas().
    all_gps_eph : list of dict
        All GPS ephemeris records.
    lla_deg_deg_m : array_like of shape (3,) or (1, 3), optional
        True receiver position [lat_deg, lon_deg, alt_m].

    Returns
    -------
    adr_resid : dict or None
        'FctSeconds', 'Svid0', 'Svid', 'ResidM';
        or None if no valid ADR data.
    """
    adr_m = gnss_meas['AdrM']
    if not np.any(np.isfinite(adr_m) & (adr_m != 0)):
        return None

    if lla_deg_deg_m is None or not np.any(lla_deg_deg_m):
        print('gps_adr_residuals needs the true position: lla_deg_deg_m')
        return None

    xyz0 = lla2xyz(np.atleast_2d(lla_deg_deg_m))[0]

    n = len(gnss_meas['FctSeconds'])
    m = len(gnss_meas['Svid'])
    week_num = np.floor(gnss_meas['FctSeconds'] / GpsConstants.WEEKSEC).astype(int)

    adr_state = gnss_meas['AdrState']

    # Choose reference satellite: the one with the most valid ADR
    num_valid_adr = np.array([
        int(np.sum((adr_state[:, j].astype(int) & (1 << 0)).astype(bool)))
        for j in range(m)
    ])
    j0 = int(np.argmax(num_valid_adr))
    svid = gnss_meas['Svid']

    adr_resid = {
        'FctSeconds': gnss_meas['FctSeconds'].copy(),
        'Svid0':      int(svid[j0]),
        'Svid':       svid.copy(),
        'ResidM':     np.full((n, m), np.nan),
    }

    # Compute expected pseudoranges at each epoch for each SV
    pr_hat_m = np.full((n, m), np.nan)
    for i in range(n):
        for j in range(m):
            ttx_sec = gnss_meas['tTxSeconds'][i, j]
            if np.isnan(ttx_sec):
                continue
            gps_eph_j, i_sv = closest_gps_eph(
                all_gps_eph, int(svid[j]), gnss_meas['FctSeconds'][i])
            if len(i_sv) == 0:
                continue
            dtsv = gps_eph2dtsv(gps_eph_j, np.array([ttx_sec]))
            ttx_true = ttx_sec - float(dtsv[0])
            sv_xyz_ttx, dtsv_out = gps_eph2xyz(
                gps_eph_j, np.array([[week_num[i], ttx_true]]))
            sv_xyz = sv_xyz_ttx[0]
            dt_flight = np.linalg.norm(xyz0 - sv_xyz) / GpsConstants.LIGHTSPEED
            sv_xyz_trx = flight_time_correction(sv_xyz, dt_flight)
            pr_hat_m[i, j] = (np.linalg.norm(xyz0 - sv_xyz_trx)
                               - GpsConstants.LIGHTSPEED * float(dtsv_out[0]))

    # Compute single-difference, then delta from t0, then residuals
    i_t0 = np.zeros(m, dtype=int)
    for i in range(n):
        if not (adr_state[i, j0].astype(int) & (1 << 0)):
            continue
        for j in range(m):
            if j == j0:
                continue
            if not (adr_state[i, j].astype(int) & (1 << 0)):
                i_t0[j] = 0
                continue
            if i_t0[j] == 0:
                if np.all(np.isfinite(pr_hat_m[i, [j, j0]])):
                    i_t0[j] = i
            i0 = i_t0[j]
            if i > i0 and np.all(np.isfinite(pr_hat_m[i, [j, j0]])):
                del_adr = ((adr_m[i, j] - adr_m[i, j0])
                           - (adr_m[i0, j] - adr_m[i0, j0]))
                del_pr_hat = ((pr_hat_m[i, j] - pr_hat_m[i, j0])
                              - (pr_hat_m[i0, j] - pr_hat_m[i0, j0]))
                resid = del_adr - del_pr_hat
                assert np.isfinite(resid), \
                    'Residual should be finite; check the above logic'
                adr_resid['ResidM'][i, j] = resid

    return adr_resid
