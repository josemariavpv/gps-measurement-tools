"""Plot C/No (PlotCno).

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


def plot_cno(gnss_meas, pr_file_name='', colors=None):
    """Plot C/No from process_gnss_meas().

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
    const_type = gnss_meas.get('ConstellationType',
                               np.ones(m, dtype=int))

    if colors is None or np.asarray(colors).shape != (m, 3):
        colors = np.zeros((m, 3))
        b_got_colors = False
    else:
        colors = np.asarray(colors, dtype=float)
        b_got_colors = True

    ax = plt.gca()
    for i in range(m):
        cn0_i = gnss_meas['Cn0DbHz'][:, i]
        i_f   = np.where(np.isfinite(cn0_i))[0]
        if len(i_f) == 0:
            continue
        lbl = sv_label(const_type[i], gnss_meas['Svid'][i])
        (h,) = ax.plot(time_s, cn0_i, label=lbl)
        color = colors[i] if b_got_colors else np.array(mcolors.to_rgb(h.get_color()))
        if not b_got_colors:
            colors[i] = color
        h.set_color(color)

        ts = lbl
        if np.isfinite(gnss_meas['AzDeg'][i]):
            ts = f'{ts}, {gnss_meas["AzDeg"][i]:03.0f}°, {gnss_meas["ElDeg"][i]:02.0f}°'
        ax.text(time_s[i_f[-1]], cn0_i[i_f[-1]], ts, color=color, fontsize=8)

    ax.legend(fontsize=7, ncol=2, loc='lower right', framealpha=0.7)

    ax.set_title('C/No in dB.Hz')
    ax.set_ylabel('(dB.Hz)')
    ax.set_xlabel(f'time (seconds)\n{pr_file_name}')
    ax.grid(True)
    plt.tight_layout()
    return colors
