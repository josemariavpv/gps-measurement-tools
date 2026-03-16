"""Weighted Least Squares PVT solver (WlsPvt).

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
from .gnss_thresholds import GnssThresholds
from .gps_eph2dtsv import gps_eph2dtsv
from .gps_eph2pvt import gps_eph2pvt
from .flight_time_correction import flight_time_correction

# Column indices in the prs matrix
J_WK  = 0
J_SEC = 1
J_SV  = 2
J_PR  = 3
J_PR_SIG = 4
J_PRR = 5
J_PRR_SIG = 6


def _check_inputs(prs, gps_eph, xo):
    """Validate WlsPvt inputs. Returns (ok, num_val)."""
    num_val = prs.shape[0]
    if prs.shape[1] != 7:
        return False, num_val
    if np.max(prs[:, J_SEC]) - np.min(prs[:, J_SEC]) > np.finfo(float).eps:
        return False, num_val
    if len(gps_eph) != num_val:
        return False, num_val
    prn_eph = np.array([e['PRN'] for e in gps_eph], dtype=int)
    if not np.all(prs[:, J_SV].astype(int) == prn_eph):
        return False, num_val
    if xo.shape != (8,):
        return False, num_val
    return True, num_val


def wls_pvt(prs, gps_eph, xo):
    """Calculate a Weighted Least Squares PVT solution.

    Parameters
    ----------
    prs : array_like, shape (n, 7)
        Each row: [trx_week, trx_sec, sv_id, pr_m, pr_sigma_m, prr_mps, prr_sigma_mps].
    gps_eph : list of dict
        GPS ephemeris records aligned with prs rows.
    xo : array_like, shape (8,)
        Initial state [x, y, z, bc, xdot, ydot, zdot, bcdot] in ECEF (m, m/s).

    Returns
    -------
    x_hat : ndarray, shape (8,)
        State update.
    z : ndarray
        A-posteriori residuals [z_pr; z_prr].
    sv_pos : ndarray, shape (n, 5)
        [sv_prn, x, y, z (m ECEF), dtsv (s)] for each satellite.
    H : ndarray, shape (n, 4)
        Observation matrix.
    wpr : ndarray, shape (n, n)
        Pseudorange weight matrix (diagonal).
    wrr : ndarray, shape (n, n)
        Pseudorange rate weight matrix (diagonal).
    """
    prs = np.atleast_2d(np.asarray(prs, dtype=float))
    xo  = np.asarray(xo, dtype=float).flatten()

    ok, num_val = _check_inputs(prs, gps_eph, xo)
    if not ok:
        raise ValueError('inputs not right size, or not properly aligned')

    x_hat = np.zeros(8)
    z = np.array([])
    H = np.zeros((num_val, 4))
    sv_pos = np.zeros((num_val, 5))

    xyz0 = xo[:3].copy()
    bc   = xo[3]

    if num_val < 4:
        return x_hat, z, sv_pos, H, np.diag(np.ones(num_val)), np.diag(np.ones(num_val))

    ttx_week    = prs[:, J_WK]
    ttx_seconds = prs[:, J_SEC] - prs[:, J_PR] / GpsConstants.LIGHTSPEED

    dtsv = gps_eph2dtsv(gps_eph, ttx_seconds)
    ttx  = ttx_seconds - dtsv  # true GPS time

    xyz_ttx, dtsv, v_mps, dtsv_dot = gps_eph2pvt(
        gps_eph, np.column_stack([ttx_week, ttx]))
    sv_xyz_trx = xyz_ttx.copy()

    wpr = np.diag(1.0 / prs[:, J_PR_SIG])
    wrr = np.diag(1.0 / prs[:, J_PRR_SIG])

    x_hat_pos = np.zeros(4)
    dx  = np.full(4, np.inf)
    while_count = 0
    max_while = 100

    while np.linalg.norm(dx) > GnssThresholds.MAXDELPOSFORNAVM:
        while_count += 1
        assert while_count < max_while, \
            f'while loop did not converge after {while_count} iterations'

        for i in range(num_val):
            dt_flight = (prs[i, J_PR] - bc) / GpsConstants.LIGHTSPEED + dtsv[i]
            sv_xyz_trx[i, :] = flight_time_correction(xyz_ttx[i, :], dt_flight)

        # line-of-sight vectors
        v_los = xyz0.reshape(3, 1) * np.ones((1, num_val)) - sv_xyz_trx.T
        rng   = np.sqrt(np.sum(v_los ** 2, axis=0))
        v_los = v_los / rng  # unit los from sv to xyz0

        sv_pos = np.column_stack([prs[:, J_SV], sv_xyz_trx, dtsv])

        pr_hat = rng + bc - GpsConstants.LIGHTSPEED * dtsv
        z_pr   = prs[:, J_PR] - pr_hat

        H = np.column_stack([v_los.T, np.ones(num_val)])
        dx = np.linalg.pinv(wpr @ H) @ (wpr @ z_pr)

        x_hat_pos += dx
        xyz0 = xyz0 + dx[:3]
        bc   = bc   + dx[3]

        z_pr = z_pr - H @ dx

    # velocity solution
    rr_mps = np.array(
        [-v_mps[i, :] @ v_los[:, i] for i in range(num_val)])
    prr_hat = rr_mps + xo[7] - GpsConstants.LIGHTSPEED * dtsv_dot
    z_prr   = prs[:, J_PRR] - prr_hat
    v_hat   = np.linalg.pinv(wrr @ H) @ (wrr @ z_prr)

    x_hat = np.concatenate([x_hat_pos, v_hat])
    z     = np.concatenate([z_pr, z_prr])

    return x_hat, z, sv_pos, H, wpr, wrr
