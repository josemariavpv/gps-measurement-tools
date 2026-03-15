"""Plot Accumulated Delta Range (PlotAdr).

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


def plot_adr(gnss_meas, pr_file_name='', colors=None):
    """Plot valid Accumulated Delta Ranges from process_adr().

    Parameters
    ----------
    gnss_meas : dict
        As returned by process_adr() (must have 'DelPrMinusAdrM').
    pr_file_name : str, optional
        Log file name shown on the x-axis label.
    colors : array_like, shape (M, 3), optional
        Per-satellite RGB colours.

    Returns
    -------
    colors : ndarray, shape (M, 3)
    """
    adr_m = gnss_meas['AdrM']
    if not np.any(np.isfinite(adr_m) & (adr_m != 0)):
        print(' No ADR to plot')
        return colors if colors is not None else np.zeros((len(gnss_meas['Svid']), 3))

    m = len(gnss_meas['Svid'])
    time_s = gnss_meas['FctSeconds'] - gnss_meas['FctSeconds'][0]

    if colors is None or np.asarray(colors).shape != (m, 3):
        colors = np.zeros((m, 3))
        b_got_colors = False
    else:
        colors = np.asarray(colors, dtype=float)
        b_got_colors = True

    ax1 = plt.subplot(5, 1, (1, 2))
    ax2 = plt.subplot(5, 1, (3, 4))
    ax3 = plt.subplot(5, 1, 5)

    for j in range(m):
        adr_j   = adr_m[:, j]
        adr_st  = gnss_meas['AdrState'][:, j]
        i_valid = (adr_st.astype(int) & (1 << 0)).astype(bool)
        i_fi    = np.where(np.isfinite(adr_j) & i_valid)[0]
        if len(i_fi) == 0:
            continue

        t_end = time_s[i_fi[-1]]
        (h,) = ax1.plot(time_s, adr_j, '.', markersize=4)
        color = colors[j] if b_got_colors else np.array(h.get_color())
        if not b_got_colors:
            colors[j] = color
        h.set_color(color)
        ax1.text(t_end, adr_j[i_fi[-1]], str(gnss_meas['Svid'][j]),
                 color=color, fontsize=8)

        ax2.plot(time_s, gnss_meas['DelPrMinusAdrM'][:, j],
                 '.', markersize=4, color=color)

    ax1.set_title('Valid Accumulated Delta Range (= -k*carrier phase) vs time')
    ax1.set_ylabel('(meters)')
    ax1.grid(True)
    ax2.set_title('DelPrM - AdrM')
    ax2.set_ylabel('(meters)')
    ax2.grid(True)

    b_clock_dis = np.concatenate([[False], np.diff(gnss_meas['ClkDCount']) != 0])
    i_cont = ~b_clock_dis
    i_dis  =  b_clock_dis
    ax3.plot(time_s[i_cont], b_clock_dis[i_cont].astype(float), '.b')
    ax3.plot(time_s[i_dis],  b_clock_dis[i_dis].astype(float),  '.r')
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['continuous', 'discontinuous'])
    ax3.grid(True)
    ax3.set_title('HW Clock Discontinuity')
    ax3.set_xlabel(f'time (seconds)\n{pr_file_name}')

    ax2.sharex(ax1)
    ax3.sharex(ax1)
    plt.tight_layout()
    return colors
