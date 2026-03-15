"""Plot pseudorange rates (PlotPseudorangeRates).

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


def plot_pseudorange_rates(gnss_meas, pr_file_name='', colors=None):
    """Plot pseudorange rates from process_gnss_meas().

    Parameters
    ----------
    gnss_meas : dict
        As returned by process_gnss_meas().
    pr_file_name : str, optional
        Log file name shown on the x-axis label.
    colors : array_like, shape (M, 3), optional
        Per-satellite RGB colours.

    Returns
    -------
    colors : ndarray, shape (M, 3)
    """
    m  = len(gnss_meas['Svid'])
    time_s = gnss_meas['FctSeconds'] - gnss_meas['FctSeconds'][0]

    if colors is None or np.asarray(colors).shape != (m, 3):
        colors = np.zeros((m, 3))
        b_got_colors = False
    else:
        colors = np.asarray(colors, dtype=float)
        b_got_colors = True

    gray = (0.5, 0.5, 0.5)
    ax1 = plt.subplot(5, 1, (1, 4))
    ax2 = plt.subplot(5, 1, 5)

    del_pr = gnss_meas['DelPrM']
    delta_mean_m = np.full(m, np.nan)

    for i in range(m):
        y_prr = gnss_meas['PrrMps'][:, i]
        i_fi  = np.where(np.isfinite(y_prr))[0]
        if len(i_fi) == 0:
            continue
        ax1.plot(time_s, y_prr, color=gray)
        ax1.text(time_s[i_fi[0]], y_prr[i_fi[0]], str(gnss_meas['Svid'][i]),
                 color=gray, ha='right', fontsize=8)
        mean_prr = float(np.mean(y_prr[i_fi]))

        y_dp = del_pr[:, i].copy()
        i_fi2 = np.where(np.isfinite(y_dp))[0]
        if len(i_fi2) > 1:
            dy = np.diff(y_dp) / np.diff(time_s)
            (h,) = ax1.plot(time_s[1:], dy, '.', markersize=4)
            color = colors[i] if b_got_colors else np.array(h.get_color())
            if not b_got_colors:
                colors[i] = color
            h.set_color(color)
            i_fi3 = np.where(np.isfinite(dy))[0]
            if len(i_fi3) > 0:
                ax1.text(time_s[1:][i_fi3[-1]], dy[i_fi3[-1]],
                         str(gnss_meas['Svid'][i]), color=color, fontsize=8)
                delta_mean_m[i] = mean_prr - float(np.mean(dy[i_fi3]))

    ts = 'diff(raw pr)/diff(time) and reported prr'
    ax1.set_title(ts)
    ax1.set_ylabel('(m/s)')

    b_clock_dis = np.concatenate([[False], np.diff(gnss_meas['ClkDCount']) != 0])
    i_cont = ~b_clock_dis
    i_dis  =  b_clock_dis
    ax2.plot(time_s[i_cont], b_clock_dis[i_cont].astype(float), '.b')
    ax2.plot(time_s[i_dis],  b_clock_dis[i_dis].astype(float),  '.r')
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['continuous', 'discontinuous'])
    ax2.grid(True)
    ax2.set_title('HW Clock Discontinuity')
    ax2.set_xlabel(f'time (seconds)\n{pr_file_name}')
    ax2.sharex(ax1)

    plt.tight_layout()
    return colors
