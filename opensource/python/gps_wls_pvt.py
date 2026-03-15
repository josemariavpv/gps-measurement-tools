"""High-level GPS WLS PVT solver (GpsWlsPvt).

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
from .closest_gps_eph import closest_gps_eph
from .wls_pvt import wls_pvt
from .xyz2lla import xyz2lla
from .rot_ecef2ned import rot_ecef2ned


def gps_wls_pvt(gnss_meas, all_gps_eph):
    """Compute PVT solution from processed GNSS measurements.

    Parameters
    ----------
    gnss_meas : dict
        Processed GNSS measurements as returned by process_gnss_meas().
    all_gps_eph : list of dict
        All GPS ephemeris records.

    Returns
    -------
    gps_pvt : dict
        Navigation solution with keys:
          'FctSeconds'    – (N,) time vector
          'allLlaDegDegM' – (N, 3) [lat_deg, lon_deg, alt_m]
          'sigmaLlaM'     – (N, 3) standard deviation of position (m)
          'allBcMeters'   – (N,) common bias (m)
          'allVelMps'     – (N, 3) velocity in NED (m/s)
          'sigmaVelMps'   – (N, 3) standard deviation of velocity (m/s)
          'allBcDotMps'   – (N,) common frequency bias (m/s)
          'numSvs'        – (N,) number of satellites used
          'hdop'          – (N,) horizontal DOP
    """
    n = len(gnss_meas['FctSeconds'])
    xo = np.zeros(8)  # initial state: centre of Earth, bc=0, v=0

    week_num = np.floor(gnss_meas['FctSeconds'] / GpsConstants.WEEKSEC).astype(int)

    gps_pvt = {
        'FctSeconds':    gnss_meas['FctSeconds'].copy(),
        'allLlaDegDegM': np.full((n, 3), np.nan),
        'sigmaLlaM':     np.full((n, 3), np.nan),
        'allBcMeters':   np.full(n, np.nan),
        'allVelMps':     np.full((n, 3), np.nan),
        'sigmaVelMps':   np.full((n, 3), np.nan),
        'allBcDotMps':   np.full(n, np.nan),
        'numSvs':        np.zeros(n, dtype=int),
        'hdop':          np.full(n, np.inf),
    }

    for i in range(n):
        pr_row = gnss_meas['PrM'][i, :]
        i_valid = np.where(np.isfinite(pr_row))[0]
        svid = gnss_meas['Svid'][i_valid]

        gps_eph, i_sv = closest_gps_eph(
            all_gps_eph, svid, gnss_meas['FctSeconds'][i])
        svid = svid[i_sv]
        num_svs = len(svid)
        gps_pvt['numSvs'][i] = num_svs
        if num_svs < 4:
            continue

        # Build prs matrix: [trx_week, trx_sec, svid, pr_m, pr_sig_m, prr_mps, prr_sig_mps]
        idx = i_valid[i_sv]
        pr_m         = gnss_meas['PrM'][i, idx]
        pr_sigma_m   = gnss_meas['PrSigmaM'][i, idx]
        prr_mps      = gnss_meas['PrrMps'][i, idx]
        prr_sigma_mps = gnss_meas['PrrSigmaMps'][i, idx]
        t_rx = np.column_stack([
            np.full(num_svs, week_num[i]),
            gnss_meas['tRxSeconds'][i, idx],
        ])
        prs = np.column_stack([t_rx, svid, pr_m, pr_sigma_m, prr_mps, prr_sigma_mps])

        xo[4:7] = 0.0
        x_hat, _, _, H, wpr, wrr = wls_pvt(prs, gps_eph, xo)
        xo = xo + x_hat

        # Extract position
        lla = xyz2lla(xo[:3].reshape(1, 3))[0]
        gps_pvt['allLlaDegDegM'][i, :] = lla
        gps_pvt['allBcMeters'][i]       = xo[3]

        # Extract velocity in NED
        re2n = rot_ecef2ned(lla[0], lla[1])
        v_ned = re2n @ xo[4:7]
        gps_pvt['allVelMps'][i, :]  = v_ned
        gps_pvt['allBcDotMps'][i]   = xo[7]

        # HDOP (using NED observation matrix)
        H_ned = np.column_stack([H[:, :3] @ re2n.T, np.ones(num_svs)])
        try:
            p_mat = np.linalg.inv(H_ned.T @ H_ned)
            gps_pvt['hdop'][i] = np.sqrt(p_mat[0, 0] + p_mat[1, 1])
        except np.linalg.LinAlgError:
            pass

        # Weighted position variance
        try:
            p_mat = np.linalg.inv(H_ned.T @ (wpr.T @ wpr) @ H_ned)
            gps_pvt['sigmaLlaM'][i, :] = np.sqrt(np.diag(p_mat[:3, :3]))
        except np.linalg.LinAlgError:
            pass

        # Weighted velocity variance
        try:
            p_mat = np.linalg.inv(H_ned.T @ (wrr.T @ wrr) @ H_ned)
            gps_pvt['sigmaVelMps'][i, :] = np.sqrt(np.diag(p_mat[:3, :3]))
        except np.linalg.LinAlgError:
            pass

    return gps_pvt
