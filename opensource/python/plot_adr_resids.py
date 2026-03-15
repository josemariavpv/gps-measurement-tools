"""Plot ADR residuals (PlotAdrResids).

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


def plot_adr_resids(adr_resid, gnss_meas, pr_file_name='', colors=None):
    """Plot ADR single-difference residuals.

    Parameters
    ----------
    adr_resid : dict or None
        As returned by gps_adr_residuals().
    gnss_meas : dict
        As returned by process_gnss_meas().
    pr_file_name : str, optional
        Log file name shown on the x-axis label.
    colors : array_like, shape (M, 3), optional
        Per-satellite RGB colours.
    """
    if (adr_resid is None
            or not np.any(np.isfinite(adr_resid['ResidM']))):
        print(' No adr residuals to plot')
        return

    k = 5  # number of satellites to show
    m = len(adr_resid['Svid'])
    time_s = adr_resid['FctSeconds'] - adr_resid['FctSeconds'][0]

    b_got_colors = (colors is not None
                    and np.asarray(colors).shape == (m, 3))

    # Find K satellites with most valid data
    num_valid = np.array([
        int(np.sum(np.isfinite(adr_resid['ResidM'][:, j])))
        for j in range(m)
    ])
    j_sorted = np.argsort(num_valid)[::-1]
    num_to_plot = min(k, m)

    axes = []
    for kk in range(num_to_plot):
        ax = plt.subplot(num_to_plot, 1, kk + 1)
        axes.append(ax)
        j_sv = j_sorted[kk]
        svid = adr_resid['Svid'][j_sv]
        (h,) = ax.plot(time_s, adr_resid['ResidM'][:, j_sv])
        ax.grid(True)

        j_gnss = np.where(gnss_meas['Svid'] == svid)[0]
        if len(j_gnss) > 0:
            jj = j_gnss[0]
            if b_got_colors:
                h.set_color(colors[jj])
            # Mark cycle slips
            adr_st   = gnss_meas['AdrState'][:, jj]
            i_cs     = np.where((adr_st.astype(int) & (1 << 2)).astype(bool))[0]
            if len(i_cs) > 0:
                ax.plot(time_s[i_cs], np.zeros(len(i_cs)), 'xk', markersize=5)

        ax.set_title(f'Svids {svid} - {adr_resid["Svid0"]}')
        ax.set_ylabel('(meters)')

    if axes:
        axes[-1].set_xlabel(f'time (seconds)\n{pr_file_name}')
        axes[0].set_title(
            f'ADR single difference residuals. No iono or tropo correction. '
            f'Svids {adr_resid["Svid"][j_sorted[0]]} - {adr_resid["Svid0"]}'
        )
        ax_lim = axes[0].axis()
        axes[0].text(ax_lim[0], ax_lim[2], ' "x" = declared cycle slip',
                     va='bottom', fontsize=8)
        for ax in axes[1:]:
            ax.sharex(axes[0])

    plt.tight_layout()
