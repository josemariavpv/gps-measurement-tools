"""Satellite skyplot – elevation vs azimuth polar plot (PlotSkyplot).

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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .sv_label import sv_label
from .closest_gps_eph import closest_gps_eph
from .gps_eph2xyz import gps_eph2xyz
from .gps_constants import GpsConstants
from .lla2xyz import lla2xyz
from .rot_ecef2ned import rot_ecef2ned


def _el_az_deg(rcv_lla_deg, sv_xyz_m):
    """Return (elevation_deg, azimuth_deg) for one satellite.

    Parameters
    ----------
    rcv_lla_deg : array_like, shape (3,)
        Receiver geodetic position [lat_deg, lon_deg, alt_m].
    sv_xyz_m : array_like, shape (3,) or (n, 3)
        Satellite ECEF position(s) in metres.

    Returns
    -------
    el_deg : ndarray, shape (n,)
    az_deg : ndarray, shape (n,)
        Elevation (degrees above horizon) and azimuth (degrees clockwise
        from North).
    """
    rcv_xyz = lla2xyz(np.atleast_2d(rcv_lla_deg))[0]       # (3,)
    los = np.atleast_2d(sv_xyz_m) - rcv_xyz                # (n, 3)
    re2n = rot_ecef2ned(rcv_lla_deg[0], rcv_lla_deg[1])    # (3, 3)
    d_ned = (re2n @ los.T).T                               # (n, 3) [N, E, D]

    horiz = np.hypot(d_ned[:, 0], d_ned[:, 1])
    el_deg = np.degrees(np.arctan2(-d_ned[:, 2], horiz))   # D is down → negate
    az_deg = np.degrees(np.arctan2(d_ned[:, 1], d_ned[:, 0]))  # atan2(E, N)
    return el_deg, az_deg


def plot_skyplot(gnss_meas, all_gps_eph, gps_pvt, pr_file_name='', colors=None):
    """Plot satellite tracks on a polar elevation-vs-azimuth (skyplot) diagram.

    The centre of the plot is the zenith (elevation 90°) and the outer ring
    is the horizon (elevation 0°).  Azimuth is measured clockwise from North.
    Each satellite is drawn as a coloured track with:

    * a circle (○) marking the first observed position, and
    * a square (□) marking the most recent position.

    Parameters
    ----------
    gnss_meas : dict
        As returned by ``process_gnss_meas()``.
    all_gps_eph : list of dict
        GPS ephemeris records, as returned by
        ``get_nasa_hourly_ephemeris()``.
    gps_pvt : dict
        WLS PVT solution, as returned by ``gps_wls_pvt()``.
    pr_file_name : str, optional
        Log-file name shown in the plot title.
    colors : array_like, shape (M, 3), optional
        Per-satellite RGB colours that match the other plots.
    """
    m = len(gnss_meas['Svid'])
    const_type = gnss_meas.get('ConstellationType', np.ones(m, dtype=int))
    n = len(gnss_meas['FctSeconds'])
    week_num = np.floor(
        gnss_meas['FctSeconds'] / GpsConstants.WEEKSEC
    ).astype(int)

    # ---- assign per-satellite colours consistent with other plots ----------
    if colors is None or np.asarray(colors).shape != (m, 3):
        cmap = plt.get_cmap('tab20')
        colors = np.array([np.array(mcolors.to_rgb(cmap(i % 20))) for i in range(m)])
    else:
        colors = np.asarray(colors, dtype=float)

    # ---- collect elevation / azimuth tracks per satellite ------------------
    el_tracks: list[list[float]] = [[] for _ in range(m)]
    az_tracks: list[list[float]] = [[] for _ in range(m)]

    for i in range(n):
        lla = gps_pvt['allLlaDegDegM'][i]
        if not np.all(np.isfinite(lla)):
            continue

        pr_row = gnss_meas['PrM'][i, :]
        i_valid = np.where(np.isfinite(pr_row))[0]
        if len(i_valid) == 0:
            continue
        svid_valid = gnss_meas['Svid'][i_valid]

        gps_eph, i_sv = closest_gps_eph(
            all_gps_eph, svid_valid, gnss_meas['FctSeconds'][i]
        )
        if not gps_eph:
            continue

        # approximate transmission time: t_rx − pseudorange / c
        idx = i_valid[np.asarray(i_sv, dtype=int)]
        pr_m = gnss_meas['PrM'][i, idx]
        trx_sec = gnss_meas['tRxSeconds'][i, idx]
        ttx_sec = trx_sec - pr_m / GpsConstants.LIGHTSPEED

        gps_time = np.column_stack([
            np.full(len(gps_eph), float(week_num[i])),
            ttx_sec,
        ])

        sv_xyz, _ = gps_eph2xyz(gps_eph, gps_time)   # (k, 3)

        el_arr, az_arr = _el_az_deg(lla, sv_xyz)

        for k in range(len(gps_eph)):
            j_m = int(i_valid[i_sv[k]])               # index in M dimension
            if el_arr[k] >= 0.0:                      # above horizon only
                el_tracks[j_m].append(float(el_arr[k]))
                az_tracks[j_m].append(float(az_arr[k]))

    # ---- draw polar plot ---------------------------------------------------
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_theta_zero_location('N')    # 0° (North) at the top
    ax.set_theta_direction(-1)         # azimuth increases clockwise

    # r = 90 - elevation  →  r=0 ≡ zenith, r=90 ≡ horizon
    ax.set_ylim(0, 90)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(['60°', '30°', '0°'], fontsize=7)
    ax.set_rlabel_position(45)

    any_plotted = False
    for j in range(m):
        if not el_tracks[j]:
            continue
        any_plotted = True

        az_rad = np.radians(np.array(az_tracks[j]))
        r_arr  = 90.0 - np.array(el_tracks[j])
        color  = tuple(colors[j])
        label  = sv_label(const_type[j], gnss_meas['Svid'][j])

        ax.plot(az_rad, r_arr, '-', linewidth=1.0, color=color, alpha=0.8)
        ax.plot(az_rad[0],  r_arr[0],  'o', markersize=4, color=color,
                zorder=3)
        ax.plot(az_rad[-1], r_arr[-1], 's', markersize=5, color=color,
                zorder=3)
        ax.text(az_rad[-1], r_arr[-1], f'  {label}', fontsize=7,
                color=color, ha='left', va='center', clip_on=True)

    if not any_plotted:
        ax.text(0, 0, 'No satellite data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')

    ax.set_title(
        f'Skyplot  ○ first  □ last\n{pr_file_name}',
        pad=20,
    )
    plt.tight_layout()
