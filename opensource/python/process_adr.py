"""Process Accumulated Delta Range (ProcessAdr).

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


def process_adr(gnss_meas):
    """Process Accumulated Delta Ranges obtained from process_gnss_meas().

    Adds ``gnss_meas['DelPrMinusAdrM']``:  ``DelPrM − AdrM``, re-initialised
    to zero at each discontinuity or reset.

    Parameters
    ----------
    gnss_meas : dict
        As returned by process_gnss_meas().

    Returns
    -------
    gnss_meas : dict
        Same dict with 'DelPrMinusAdrM' added.
    """
    adr_m     = gnss_meas['AdrM']
    adr_state = gnss_meas['AdrState']

    has_valid = np.any(np.isfinite(adr_m) & (adr_m != 0))
    if not has_valid:
        print(' No ADR recorded')
        return gnss_meas

    n, m = adr_m.shape
    del_pr_minus_adr = np.full((n, m), np.nan)

    for j in range(m):
        a = adr_m[:, j].copy()
        d = gnss_meas['DelPrM'][:, j].copy()
        s = adr_state[:, j]

        # GNSS_ADR_STATE_VALID  = bit 0
        # GNSS_ADR_STATE_RESET  = bit 1
        i_valid = (s.astype(int) & (1 << 0)).astype(bool)
        i_reset = (s.astype(int) & (1 << 1)).astype(bool)
        a[~i_valid] = np.nan

        del_pr_m0 = np.nan
        for i in range(n):
            if (np.isfinite(a[i]) and a[i] != 0
                    and np.isfinite(d[i]) and not i_reset[i]):
                if np.isnan(del_pr_m0):
                    del_pr_m0 = d[i] - a[i]
            else:
                del_pr_m0 = np.nan

            del_pr_minus_adr[i, j] = d[i] - del_pr_m0 - a[i]

    gnss_meas['DelPrMinusAdrM'] = del_pr_minus_adr
    return gnss_meas
