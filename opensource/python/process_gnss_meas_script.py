"""Python equivalent of ProcessGnssMeasScript.m.

Read a GnssLogger output file, compute and plot pseudoranges, C/No,
and a Weighted Least Squares PVT solution.

Usage
-----
    python process_gnss_meas_script.py

Edit the ``DIR_NAME``, ``PR_FILE_NAME``, and ``PARAM`` constants below
to point to your local data, or import and call run() from another script.

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

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; change to 'TkAgg' etc. if desired
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration – edit these before running
# -------------------------------------------------------------------------
# Demo file with duty cycling, no carrier phase:
PR_FILE_NAME = 'pseudoranges_log_2016_06_30_21_26_07.txt'
# Demo file with carrier phase:
# PR_FILE_NAME = 'pseudoranges_log_2016_08_22_14_45_50.txt'

# Directory that contains PR_FILE_NAME and where ephemeris will be cached:
DIR_NAME = os.path.join(os.path.dirname(__file__), '..', 'demoFiles')

PARAM = {
    'llaTrueDegDegM': [37.422578, -122.081678, -28],  # Charleston Park Test Site
    # 'llaTrueDegDegM': [],  # set to [] if true position is unknown
}


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def run(dir_name=DIR_NAME, pr_file_name=PR_FILE_NAME, param=None):
    """Run the full GNSS processing pipeline.

    Parameters
    ----------
    dir_name : str
        Directory containing the log file and ephemeris cache.
    pr_file_name : str
        GnssLogger log file name.
    param : dict, optional
        Processing parameters.  May contain ``'llaTrueDegDegM'``.
    """
    if param is None:
        param = PARAM

    from .set_data_filter import set_data_filter
    from .read_gnss_logger import read_gnss_logger
    from .gps2utc import gps2utc
    from .get_nasa_hourly_ephemeris import get_nasa_hourly_ephemeris
    from .process_gnss_meas import process_gnss_meas
    from .gps_wls_pvt import gps_wls_pvt
    from .process_adr import process_adr
    from .gps_adr_residuals import gps_adr_residuals
    from .plot_pseudoranges import plot_pseudoranges
    from .plot_pseudorange_rates import plot_pseudorange_rates
    from .plot_cno import plot_cno
    from .plot_pvt import plot_pvt
    from .plot_pvt_states import plot_pvt_states
    from .plot_adr import plot_adr
    from .plot_adr_resids import plot_adr_resids

    # ---- Read log file -------------------------------------------------------
    data_filter = set_data_filter()
    gnss_raw, gnss_analysis = read_gnss_logger(dir_name, pr_file_name, data_filter)
    if gnss_raw is None:
        print('read_gnss_logger returned None – check file and data filter.')
        return

    # ---- Get ephemeris -------------------------------------------------------
    fct_seconds = float(gnss_raw['allRxMillis'][-1]) * 1e-3
    utc_time = gps2utc(fct_seconds=fct_seconds)[0]
    all_gps_eph, _ = get_nasa_hourly_ephemeris(utc_time, dir_name)
    if not all_gps_eph:
        print('No GPS ephemeris obtained.')
        return

    # ---- Process raw measurements -------------------------------------------
    gnss_meas = process_gnss_meas(gnss_raw)

    # ---- Plot pseudoranges and rates ----------------------------------------
    fig1 = plt.figure()
    colors = plot_pseudoranges(gnss_meas, pr_file_name)
    fig1.savefig(os.path.join(dir_name, 'pseudoranges.png'), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure()
    plot_pseudorange_rates(gnss_meas, pr_file_name, colors)
    fig2.savefig(os.path.join(dir_name, 'pseudorange_rates.png'), dpi=150)
    plt.close(fig2)

    fig3 = plt.figure()
    plot_cno(gnss_meas, pr_file_name, colors)
    fig3.savefig(os.path.join(dir_name, 'cno.png'), dpi=150)
    plt.close(fig3)

    # ---- WLS PVT ------------------------------------------------------------
    gps_pvt = gps_wls_pvt(gnss_meas, all_gps_eph)

    # ---- Plot PVT -----------------------------------------------------------
    fig4 = plt.figure()
    ts = 'Raw Pseudoranges, Weighted Least Squares solution'
    plot_pvt(gps_pvt, pr_file_name, param.get('llaTrueDegDegM'), ts)
    fig4.savefig(os.path.join(dir_name, 'pvt.png'), dpi=150)
    plt.close(fig4)

    fig5 = plt.figure()
    plot_pvt_states(gps_pvt, pr_file_name)
    fig5.savefig(os.path.join(dir_name, 'pvt_states.png'), dpi=150)
    plt.close(fig5)

    # ---- ADR ----------------------------------------------------------------
    adr_m = gnss_meas['AdrM']
    if np.any(np.isfinite(adr_m) & (adr_m != 0)):
        gnss_meas = process_adr(gnss_meas)

        fig6 = plt.figure()
        plot_adr(gnss_meas, pr_file_name, colors)
        fig6.savefig(os.path.join(dir_name, 'adr.png'), dpi=150)
        plt.close(fig6)

        adr_resid = gps_adr_residuals(
            gnss_meas, all_gps_eph,
            param.get('llaTrueDegDegM'))

        fig7 = plt.figure()
        plot_adr_resids(adr_resid, gnss_meas, pr_file_name, colors)
        fig7.savefig(os.path.join(dir_name, 'adr_resids.png'), dpi=150)
        plt.close(fig7)

    return gnss_meas, gps_pvt


if __name__ == '__main__':
    run()
