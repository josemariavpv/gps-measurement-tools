"""Plot PVT states (PlotPvtStates).

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

from .gps_constants import GpsConstants
from .lla2ned import lla2ned


def plot_pvt_states(gnss_pvt, pr_file_name=''):
    """Plot the position, velocity and clock state time series.

    Parameters
    ----------
    gnss_pvt : dict
        As returned by gps_wls_pvt().
    pr_file_name : str, optional
        Log file name shown on the x-axis label.
    """
    i_fi = np.where(np.isfinite(gnss_pvt['allLlaDegDegM'][:, 0]))[0]
    if len(i_fi) == 0:
        return

    lla_med = np.nanmedian(gnss_pvt['allLlaDegDegM'], axis=0)
    t_s = gnss_pvt['FctSeconds'] - gnss_pvt['FctSeconds'][0]

    # --- subplot 1: NED offsets from median ---
    ax1 = plt.subplot(4, 1, 1)
    ned = lla2ned(gnss_pvt['allLlaDegDegM'], lla_med)
    ax1.plot(t_s, ned[:, 0], 'r', label='Lat')
    ax1.plot(t_s, ned[:, 1], 'g', label='Lon')
    ax1.plot(t_s, ned[:, 2], 'b', label='Alt')

    i_fi2 = np.where(np.isfinite(ned[:, 0]))[0]
    if len(i_fi2) > 0:
        for col, comp, lbl in zip(('r', 'g', 'b'),
                                   (0, 1, 2),
                                   ('Lat', 'Lon', 'Alt')):
            ax1.text(t_s[-1], ned[i_fi2[-1], comp], lbl, color=col, fontsize=8)

    ax1.set_title('WLS: Position states offset from medians [Lat,Lon,Alt]')
    ax1.grid(True)
    ax1.set_ylabel('(meters)')

    # --- subplot 2: common clock bias ---
    ax2 = plt.subplot(4, 1, 2)
    i_fi3 = np.where(np.isfinite(gnss_pvt['allBcMeters']))[0]
    if len(i_fi3) > 0:
        bc0 = gnss_pvt['allBcMeters'][i_fi3[0]]
        ax2.plot(t_s, gnss_pvt['allBcMeters'] - bc0)
        ax2.grid(True)
        bc0_us = bc0 / GpsConstants.LIGHTSPEED * 1e6
        ax2.set_title(
            f'Common bias "clock" offset from initial value of {bc0_us:.1f} µs')
        ax2.set_ylabel('meters')

        ax2r = ax2.twinx()
        bc_range = ax2.get_ylim()
        ax2r.set_ylim(np.array(bc_range) / GpsConstants.LIGHTSPEED * 1e6)
        ax2r.set_ylabel('(microseconds)')

    # --- subplot 3: velocity ---
    ax3 = plt.subplot(4, 1, 3)
    vel = gnss_pvt['allVelMps']
    ax3.plot(t_s, vel[:, 0], 'r', label='North')
    ax3.plot(t_s, vel[:, 1], 'g', label='East')
    ax3.plot(t_s, vel[:, 2], 'b', label='Down')

    i_fi4 = np.where(np.isfinite(vel[:, 0]))[0]
    if len(i_fi4) > 0:
        for col, comp, lbl in zip(('r', 'g', 'b'),
                                   (0, 1, 2),
                                   ('North', 'East', 'Down')):
            ax3.text(t_s[-1], vel[i_fi4[-1], comp], lbl, color=col, fontsize=8)

    ax3.set_title('Velocity states [North,East,Down]')
    ax3.grid(True)
    ax3.set_ylabel('(m/s)')

    # --- subplot 4: frequency bias ---
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(t_s, gnss_pvt['allBcDotMps'])
    ax4.grid(True)
    ax4.set_title('Common frequency offset')
    ax4.set_ylabel('(m/s)')
    ax4.set_xlabel(f'\ntime(seconds)\n{pr_file_name}')

    ppm_per_mps = 1.0 / GpsConstants.LIGHTSPEED * 1e6
    ax4r = ax4.twinx()
    ax4r.set_ylim(np.array(ax4.get_ylim()) * ppm_per_mps)
    ax4r.set_ylabel('(ppm)')

    # Link all x-axes
    for ax in [ax2, ax3, ax4]:
        ax.sharex(ax1)

    plt.tight_layout()
