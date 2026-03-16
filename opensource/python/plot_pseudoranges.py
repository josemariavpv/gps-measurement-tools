"""Plot pseudoranges (PlotPseudoranges).

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


def plot_pseudoranges(gnss_meas, pr_file_name='', colors=None):
    """Plot pseudoranges from process_gnss_meas().

    Parameters
    ----------
    gnss_meas : dict
        As returned by process_gnss_meas().
    pr_file_name : str, optional
        Log file name shown on the x-axis label.
    colors : array_like, shape (M, 3), optional
        Per-satellite RGB colours; ignored if wrong shape.

    Returns
    -------
    colors : ndarray, shape (M, 3)
        Colour matrix (one row per satellite).
    """
    m = len(gnss_meas['Svid'])
    n = len(gnss_meas['FctSeconds'])
    time_s = gnss_meas['FctSeconds'] - gnss_meas['FctSeconds'][0]
    const_type = gnss_meas.get('ConstellationType',
                               np.ones(m, dtype=int))

    if colors is None or np.asarray(colors).shape != (m, 3):
        colors = np.zeros((m, 3))
        b_got_colors = False
    else:
        colors = np.asarray(colors, dtype=float)
        b_got_colors = True

    fig = plt.gcf()
    ax1 = plt.subplot(5, 1, (1, 2))
    ax2 = plt.subplot(5, 1, (3, 4))
    ax3 = plt.subplot(5, 1, 5)

    for i in range(m):
        pr_i = gnss_meas['PrM'][:, i]
        i_f  = np.where(np.isfinite(pr_i))[0]
        if len(i_f) == 0:
            continue
        t_end = time_s[i_f[-1]]
        lbl = sv_label(const_type[i], gnss_meas['Svid'][i])
        (h,) = ax1.plot(time_s, pr_i, '.', markersize=4, label=lbl)
        color = colors[i] if b_got_colors else np.array(mcolors.to_rgb(h.get_color()))
        if not b_got_colors:
            colors[i] = color
        h.set_color(color)
        ax1.text(t_end, pr_i[i_f[-1]], lbl, color=color, fontsize=8)

        y = pr_i - pr_i[i_f[0]]
        (h2,) = ax2.plot(time_s, y, '.', markersize=4, color=color, label=lbl)
        if len(i_f) > 0:
            ax2.text(t_end, y[i_f[-1]], lbl, color=color, fontsize=8)

    ax1.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.7)

    ax1.set_title('Pseudoranges vs time')
    ax1.set_ylabel('(meters)')
    ax2.set_title('Pseudoranges change from initial value')
    ax2.set_ylabel('(meters)')

    b_clock_dis = np.concatenate([[False], np.diff(gnss_meas['ClkDCount']) != 0])
    i_cont   = ~b_clock_dis
    i_dis    = b_clock_dis
    ax3.plot(time_s[i_cont],  b_clock_dis[i_cont].astype(float),  '.b')
    ax3.plot(time_s[i_dis],   b_clock_dis[i_dis].astype(float),   '.r')
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['continuous', 'discontinuous'])
    ax3.grid(True)
    ax3.set_title('HW Clock Discontinuity')
    ax3.set_xlabel(f'time (seconds)\n{pr_file_name}')

    # Link x-axes
    ax2.sharex(ax1)
    ax3.sharex(ax1)

    plt.tight_layout()
    return colors
