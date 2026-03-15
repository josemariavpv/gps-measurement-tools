"""Process raw GNSS measurements from GnssLogger (ProcessGnssMeas).

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
from .gps_constants import GpsConstants
from .gnss_thresholds import GnssThresholds


def process_gnss_meas(gnss_raw):
    """Process raw GNSS measurements read from read_gnss_logger().

    Parameters
    ----------
    gnss_raw : dict
        Output from read_gnss_logger().

    Returns
    -------
    gnss_meas : dict
        Processed measurements with keys:
          'FctSeconds'    – (N,) full-cycle time tag (seconds)
          'ClkDCount'     – (N,) hardware clock discontinuity count
          'HwDscDelS'     – (N,) clock change during each discontinuity (s)
          'Svid'          – (M,) all SV IDs found in gnss_raw
          'AzDeg'         – (M,) azimuth (NaN, filled later)
          'ElDeg'         – (M,) elevation (NaN, filled later)
          'tRxSeconds'    – (N, M) time of reception (GPS seconds of week)
          'tTxSeconds'    – (N, M) time of transmission (GPS seconds of week)
          'PrM'           – (N, M) pseudoranges (m)
          'PrSigmaM'      – (N, M) pseudorange std dev (m)
          'DelPrM'        – (N, M) change in PR while clock is continuous
          'PrrMps'        – (N, M) pseudorange rate (m/s)
          'PrrSigmaMps'   – (N, M) PR rate std dev (m/s)
          'AdrM'          – (N, M) accumulated delta range (m)
          'AdrSigmaM'     – (N, M) ADR std dev (m)
          'AdrState'      – (N, M) ADR state flags
          'Cn0DbHz'       – (N, M) carrier-to-noise (dB-Hz)
    """
    gnss_raw = _filter_valid(gnss_raw)

    all_rx_ms = gnss_raw['allRxMillis'].astype(float)
    fct_seconds = np.unique(all_rx_ms) * 1e-3
    n = len(fct_seconds)
    svid_all = np.unique(gnss_raw['Svid'].astype(int))
    m = len(svid_all)

    gnss_meas = {
        'FctSeconds':    fct_seconds,
        'ClkDCount':     np.zeros(n, dtype=int),
        'HwDscDelS':     np.zeros(n),
        'Svid':          svid_all,
        'AzDeg':         np.full(m, np.nan),
        'ElDeg':         np.full(m, np.nan),
        'tRxSeconds':    np.full((n, m), np.nan),
        'tTxSeconds':    np.full((n, m), np.nan),
        'PrM':           np.full((n, m), np.nan),
        'PrSigmaM':      np.full((n, m), np.nan),
        'DelPrM':        np.full((n, m), np.nan),
        'PrrMps':        np.full((n, m), np.nan),
        'PrrSigmaMps':   np.full((n, m), np.nan),
        'AdrM':          np.full((n, m), np.nan),
        'AdrSigmaM':     np.full((n, m), np.nan),
        'AdrState':      np.zeros((n, m)),
        'Cn0DbHz':       np.full((n, m), np.nan),
    }

    # GPS week number
    week_number = np.floor(
        -gnss_raw['FullBiasNanos'].astype(np.int64).astype(float) * 1e-9
        / GpsConstants.WEEKSEC).astype(np.int64)

    bias_nanos = gnss_raw.get('BiasNanos',       np.zeros(len(gnss_raw['TimeNanos'])))
    toff_nanos = gnss_raw.get('TimeOffsetNanos',  np.zeros(len(gnss_raw['TimeNanos'])))

    WEEKNANOS = np.int64(GpsConstants.WEEKSEC * 1e9)
    week_number_nanos = week_number * WEEKNANOS

    # tRxNanos since start of week, using FullBiasNanos[0] to track drift
    t_rx_nanos = (gnss_raw['TimeNanos'].astype(np.int64)
                  - gnss_raw['FullBiasNanos'].astype(np.int64)[0]
                  - week_number_nanos)

    state0 = int(gnss_raw['State'][0])
    assert (state0 & (1 << 0)) and (state0 & (1 << 3)), \
        ('gnssRaw.State[0] must have bits 0 and 3 true before calling '
         'process_gnss_meas')

    assert np.all(t_rx_nanos >= 0), 'tRxNanos should be >= 0'

    t_rx_seconds = (t_rx_nanos.astype(float) - toff_nanos - bias_nanos) * 1e-9
    t_tx_seconds = gnss_raw['ReceivedSvTimeNanos'].astype(float) * 1e-9

    pr_seconds, t_rx_seconds = _check_gps_week_rollover(t_rx_seconds, t_tx_seconds)
    pr_m       = pr_seconds * GpsConstants.LIGHTSPEED
    pr_sigma_m = gnss_raw['ReceivedSvTimeUncertaintyNanos'].astype(float) * 1e-9 \
                 * GpsConstants.LIGHTSPEED
    prr_mps      = gnss_raw['PseudorangeRateMetersPerSecond'].astype(float)
    prr_sigma    = gnss_raw['PseudorangeRateUncertaintyMetersPerSecond'].astype(float)
    adr_m        = gnss_raw['AccumulatedDeltaRangeMeters'].astype(float)
    adr_sigma    = gnss_raw['AccumulatedDeltaRangeUncertaintyMeters'].astype(float)
    adr_state    = gnss_raw['AccumulatedDeltaRangeState'].astype(float)
    cn0          = gnss_raw['Cn0DbHz'].astype(float)
    svid_raw     = gnss_raw['Svid'].astype(int)
    hw_count     = gnss_raw['HardwareClockDiscontinuityCount'].astype(int)

    for i in range(n):
        j_idx = np.where(np.abs(fct_seconds[i] * 1e3 - all_rx_ms) < 1)[0]
        for jj in j_idx:
            k = int(np.where(svid_all == svid_raw[jj])[0][0])
            gnss_meas['tRxSeconds'][i, k]  = t_rx_seconds[jj]
            gnss_meas['tTxSeconds'][i, k]  = t_tx_seconds[jj]
            gnss_meas['PrM'][i, k]         = pr_m[jj]
            gnss_meas['PrSigmaM'][i, k]    = pr_sigma_m[jj]
            gnss_meas['PrrMps'][i, k]      = prr_mps[jj]
            gnss_meas['PrrSigmaMps'][i, k] = prr_sigma[jj]
            gnss_meas['AdrM'][i, k]        = adr_m[jj]
            gnss_meas['AdrSigmaM'][i, k]   = adr_sigma[jj]
            gnss_meas['AdrState'][i, k]    = adr_state[jj]
            gnss_meas['Cn0DbHz'][i, k]     = cn0[jj]
        gnss_meas['ClkDCount'][i] = hw_count[j_idx[0]]
        if hw_count[j_idx[0]] != hw_count[j_idx[-1]]:
            raise ValueError(
                'HardwareClockDiscontinuityCount changed within the same epoch')

    gnss_meas = _get_del_pr(gnss_meas)
    return gnss_meas


# -------------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------------

def _filter_valid(gnss_raw):
    """Remove measurements with large uncertainties."""
    tow_unc  = gnss_raw['ReceivedSvTimeUncertaintyNanos'].astype(float)
    prr_unc  = gnss_raw['PseudorangeRateUncertaintyMetersPerSecond'].astype(float)

    i_tow = tow_unc  > GnssThresholds.MAXTOWUNCNS
    i_prr = prr_unc  > GnssThresholds.MAXPRRUNCMPS
    i_bad = i_tow | i_prr

    if np.any(i_bad):
        num_bad = int(np.sum(i_bad))
        if num_bad >= len(i_bad):
            raise AssertionError('Removing all measurements in gnss_raw')
        print(f'\nRemoved {num_bad} bad meas inside process_gnss_meas > '
              f'filter_valid because:')
        if np.any(i_tow):
            print(f'  towUnc > {GnssThresholds.MAXTOWUNCNS:.0f} ns')
        if np.any(i_prr):
            print(f'  prrUnc > {GnssThresholds.MAXPRRUNCMPS:.0f} m/s')
        good = ~i_bad
        gnss_raw = {k: v[good] for k, v in gnss_raw.items()}

    return gnss_raw


def _check_gps_week_rollover(t_rx_seconds, t_tx_seconds):
    """Detect and correct GPS week rollover in time-of-reception."""
    pr_seconds = t_rx_seconds - t_tx_seconds
    i_roll = pr_seconds > GpsConstants.WEEKSEC / 2

    if np.any(i_roll):
        print('\nWARNING: week rollover detected in time tags. Adjusting ...')
        pr_s = pr_seconds[i_roll]
        del_s = np.round(pr_s / GpsConstants.WEEKSEC) * GpsConstants.WEEKSEC
        pr_s  = pr_s - del_s
        max_bias = 10.0
        if np.any(pr_s > max_bias):
            raise ValueError('Failed to correct week rollover')
        pr_seconds[i_roll]    = pr_s
        t_rx_seconds[i_roll] -= del_s
        print('Corrected week rollover')

    return pr_seconds, t_rx_seconds


def _get_del_pr(gnss_meas):
    """Compute DelPrM: change in pseudorange while clock is continuous."""
    n, m = gnss_meas['PrM'].shape
    b_clock_dis = np.concatenate(
        [[False], np.diff(gnss_meas['ClkDCount']) != 0])

    del_pr = np.zeros((n, m))
    del_pr[1:, :] = np.nan

    for j in range(m):
        i0 = 0
        for i in range(1, n):
            if b_clock_dis[i] or np.isnan(gnss_meas['PrM'][i0, j]):
                i0 = i
            if b_clock_dis[i]:
                del_pr[i, j] = np.nan
            else:
                del_pr[i, j] = gnss_meas['PrM'][i, j] - gnss_meas['PrM'][i0, j]

    gnss_meas['DelPrM'] = del_pr
    return gnss_meas
