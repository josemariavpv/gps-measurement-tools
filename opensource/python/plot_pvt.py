"""Plot PVT solution (PlotPvt).

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .lla2ned import lla2ned


def plot_pvt(gps_pvt, pr_file_name='', lla_true_deg_deg_m=None,
             title_string='PVT solution'):
    """Plot the WLS PVT results.

    Parameters
    ----------
    gps_pvt : dict
        As returned by gps_wls_pvt().
    pr_file_name : str, optional
        Log file name shown on the x-axis label.
    lla_true_deg_deg_m : array_like of shape (3,), optional
        True position for reference [lat_deg, lon_deg, alt_m].
    title_string : str, optional
        Plot title.
    """
    gray   = (0.5, 0.5, 0.5)
    ltgray = (0.85, 0.85, 0.85)

    i_fi = np.where(np.isfinite(gps_pvt['allLlaDegDegM'][:, 0]))[0]
    if len(i_fi) == 0:
        return

    lla_med = np.nanmedian(gps_pvt['allLlaDegDegM'], axis=0)
    print(f'Median llaDegDegM = [{lla_med[0]:.7f} {lla_med[1]:.7f} {lla_med[2]:.2f}]')

    b_got_true = (lla_true_deg_deg_m is not None
                  and np.any(lla_true_deg_deg_m))
    lla_ref = np.asarray(lla_true_deg_deg_m) if b_got_true else lla_med

    ned_m = lla2ned(gps_pvt['allLlaDegDegM'], lla_ref)
    t_s   = gps_pvt['FctSeconds'] - gps_pvt['FctSeconds'][0]

    # --- subplot 1: NE scatter ---
    ax1 = plt.subplot(4, 1, (1, 2))
    ax1.plot(ned_m[:, 1], ned_m[:, 0], '-', linewidth=0.3, color=ltgray)
    ax1.plot(ned_m[:, 1], ned_m[:, 0], 'cx', markersize=4)

    ned_med = lla2ned(lla_med.reshape(1, 3), lla_ref)[0]
    ax1.plot(ned_med[1], ned_med[0], '+k', markersize=12)
    ax1.text(ned_med[1], ned_med[0],
             f'  median [{lla_med[0]:.6f}°, {lla_med[1]:.6f}°]', color='k')

    if b_got_true:
        ax1.plot(0, 0, '+r', markersize=12)
        ax1.text(0, 0,
                 f' true pos [{lla_ref[0]:.6f}°, {lla_ref[1]:.6f}°]', color='r')
        err_m = np.linalg.norm(ned_med[:2])
        ax1.text(min(0, ned_med[1]), np.nanmin(ned_m[:, 0]),
                 f'|median - true pos| = {err_m:.1f} m',
                 va='bottom', color='k')

    ax1.set_title(title_string)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_ylabel('North (m)')
    ax1.set_xlabel('East (m)')

    dist_m = np.sqrt(np.sum(ned_m[i_fi, :2] ** 2, axis=1))
    med_m  = float(np.median(dist_m))
    circ   = plt.Circle((0, 0), med_m, fill=False, edgecolor=gray)
    ax1.add_patch(circ)
    ax1.text(0, med_m, f'50% distribution = {med_m:.1f} m',
             va='bottom', color=gray)

    # --- subplot 2: speed ---
    ax2 = plt.subplot(4, 1, 3)
    i_good = np.isfinite(gps_pvt['allVelMps'][:, 0])
    speed_mps = np.full(len(t_s), np.nan)
    speed_mps[i_good] = np.sqrt(np.sum(gps_pvt['allVelMps'][i_good, :2] ** 2,
                                       axis=1))
    ax2.plot(t_s, speed_mps)
    ax2.grid(True)
    ax2.set_ylabel('Horiz. speed (m/s)')

    # --- subplot 3: HDOP + # sats ---
    ax3 = plt.subplot(4, 1, 4)
    ax3.plot(t_s, gps_pvt['hdop'], label='HDOP')
    ax3b = ax3.twinx()
    ax3b.step(t_s, gps_pvt['numSvs'], 'g', where='post', linewidth=1.5,
              label='# sats')
    ax3.set_ylabel('HDOP')
    ax3b.set_ylabel('# sats')
    ax3.grid(True)
    ax3.set_xlabel(f'time(seconds)\n{pr_file_name}')

    ax2.sharex(ax3)
    plt.tight_layout()
