"""RINEX observation quality analysis plots.

Four stand-alone plot functions for diagnosing GNSS observation files:

* :func:`plot_rinex_visibility`   – satellite tracking timeline
* :func:`plot_rinex_availability` – observable-availability heatmap
* :func:`plot_rinex_cn0`          – signal strength (C/N0) over time
* :func:`plot_rinex_cycle_slips`  – cycle-slip detections from LLI flag

Each function accepts the dict returned by :func:`read_rinex_obs` and an
optional *file_name* string that is appended to the x-axis label.

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
import matplotlib.colors as mcolors

from .read_rinex_obs import get_signal_strength


# ---------------------------------------------------------------------------
# Constellation colour palette (consistent across all four plots)
# ---------------------------------------------------------------------------

_SYS_COLOR = {
    'G': '#1F77B4',  # GPS       – blue
    'R': '#FF7F0E',  # GLONASS   – orange
    'E': '#2CA02C',  # Galileo   – green
    'C': '#D62728',  # BeiDou    – red
    'J': '#9467BD',  # QZSS      – purple
    'S': '#8C564B',  # SBAS      – brown
    'I': '#E377C2',  # NavIC     – pink
    '?': '#7F7F7F',  # unknown   – grey
}

_SYS_NAME = {
    'G': 'GPS',
    'R': 'GLONASS',
    'E': 'Galileo',
    'C': 'BeiDou',
    'J': 'QZSS',
    'S': 'SBAS',
    'I': 'NavIC',
    '?': 'Unknown',
}


def _sys(sat_id):
    return sat_id[0] if sat_id else '?'


def _color(sat_id):
    return _SYS_COLOR.get(_sys(sat_id), _SYS_COLOR['?'])


def _xlabel(time_s, file_name=''):
    """Return a consistent x-axis label."""
    dur_min = time_s[-1] / 60.0 if len(time_s) > 1 else 0.0
    lbl = f'time (s)  – duration {dur_min:.0f} min'
    if file_name:
        lbl += f'\n{file_name}'
    return lbl


# ===========================================================================
# 1. Visibility timeline
# ===========================================================================

def plot_rinex_visibility(rinex_obs, file_name=''):
    """Plot a satellite-tracking timeline (gantt chart).

    Each satellite is shown as a horizontal bar coloured by constellation.
    Gaps in tracking appear as breaks in the bar.

    Parameters
    ----------
    rinex_obs : dict
        As returned by :func:`read_rinex_obs`.
    file_name : str, optional
        Source file name, appended to the x-axis label.
    """
    times = rinex_obs['times']
    sats  = rinex_obs['sats']
    data  = rinex_obs['data']

    if not sats or len(times) == 0:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=plt.gca().transAxes)
        return

    fig = plt.gcf()
    fig.clf()
    ax  = fig.add_subplot(111)

    # Determine which obs code to use as "presence" indicator
    # Priority: C1* (pseudorange), L1* (phase), any code
    def _presence(sv):
        if sv not in data:
            return np.zeros(len(times), dtype=bool)
        codes = list(data[sv].keys())
        for prefix in ('C1', 'L1', 'C2', 'L2', 'C5', 'L5'):
            for c in codes:
                if c.startswith(prefix):
                    arr = data[sv][c]
                    return np.isfinite(arr)
        # fallback: any code with data
        for c in codes:
            mask = np.isfinite(data[sv][c])
            if np.any(mask):
                return mask
        return np.zeros(len(times), dtype=bool)

    # Sort satellites: GPS first, then alphabetically
    _order = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4, 'S': 5, 'I': 6, '?': 7}
    sorted_sats = sorted(sats, key=lambda s: (_order.get(s[0], 9), s))

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0

    for k, sv in enumerate(sorted_sats):
        pres = _presence(sv)
        color = _color(sv)
        y = len(sorted_sats) - 1 - k  # top = first satellite

        # Draw bar segments
        i_on = np.where(np.diff(pres.astype(int)) == 1)[0] + 1
        i_off = np.where(np.diff(pres.astype(int)) == -1)[0] + 1
        if pres[0]:
            i_on = np.concatenate([[0], i_on])
        if pres[-1]:
            i_off = np.concatenate([i_off, [len(pres)]])
        for i_s, i_e in zip(i_on, i_off):
            t_start = times[i_s]
            t_end   = times[i_e - 1] + dt
            ax.barh(y, t_end - t_start, left=t_start, height=0.7,
                    color=color, alpha=0.85, linewidth=0)

    # Y-axis labels
    ax.set_yticks(range(len(sorted_sats)))
    ax.set_yticklabels(list(reversed(sorted_sats)), fontsize=8)
    ax.set_ylim(-0.5, len(sorted_sats) - 0.5)
    ax.set_xlim(times[0], times[-1] + dt)
    ax.set_xlabel(_xlabel(times, file_name), fontsize=8)
    ax.set_title('Satellite Visibility Timeline')
    ax.grid(True, axis='x', alpha=0.3)

    # Constellation legend
    seen_sys = {s[0] for s in sorted_sats}
    patches = [
        mpatches.Patch(color=_SYS_COLOR.get(sc, '#888'),
                       label=_SYS_NAME.get(sc, sc))
        for sc in sorted('GREJCSI', key=lambda x: _order.get(x, 9))
        if sc in seen_sys
    ]
    if patches:
        ax.legend(handles=patches, fontsize=8, loc='lower right',
                  framealpha=0.8)

    # Satellite count annotation
    n_sats = len(sorted_sats)
    n_sys  = len(seen_sys)
    ax.set_title(
        f'Satellite Visibility Timeline  '
        f'({n_sats} satellites, {n_sys} constellation{"s" if n_sys != 1 else ""})'
    )
    fig.tight_layout()


# ===========================================================================
# 2. Observable availability heatmap
# ===========================================================================

def plot_rinex_availability(rinex_obs, file_name=''):
    """Plot a heatmap of observable availability (presence / gap map).

    Each row is a (satellite, obs-code) pair; each column is an epoch.  A
    coloured cell means the observation is present; grey means absent.

    Parameters
    ----------
    rinex_obs : dict
        As returned by :func:`read_rinex_obs`.
    file_name : str, optional
        Source file name.
    """
    times = rinex_obs['times']
    sats  = rinex_obs['sats']
    data  = rinex_obs['data']

    if not sats or len(times) == 0:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=plt.gca().transAxes)
        return

    fig = plt.gcf()
    fig.clf()

    # Build the matrix rows: one row per (sat, obs_code)
    _order = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4, 'S': 5, 'I': 6, '?': 7}
    sorted_sats = sorted(sats, key=lambda s: (_order.get(s[0], 9), s))

    # Limit to "important" codes: pseudorange and phase at L1/L2/L5
    _prio = ('C1', 'L1', 'C2', 'L2', 'C5', 'L5', 'S1', 'S2', 'S5')

    rows   = []   # (label, ndarray boolean presence, color)
    for sv in sorted_sats:
        if sv not in data:
            continue
        codes = sorted(data[sv].keys())
        # Sort by priority, then alphabetically
        def _sort_key(c):
            for k, p in enumerate(_prio):
                if c.startswith(p):
                    return (k, c)
            return (len(_prio), c)
        codes = sorted(codes, key=_sort_key)
        for code in codes:
            arr  = data[sv][code]
            pres = np.isfinite(arr).astype(float)
            # Use NaN where absent so imshow shows it differently
            mat  = np.where(pres, 1.0, np.nan)
            rows.append((f'{sv} {code}', mat, _color(sv)))

    if not rows:
        plt.text(0.5, 0.5, 'No observable data', ha='center', va='center',
                 transform=plt.gca().transAxes)
        return

    # Downsample in time if too many epochs (> 2000 columns → pixelate)
    n_cols = len(times)
    step   = max(1, n_cols // 1000)
    t_ds   = times[::step]
    matrix = np.vstack([r[1][::step] for r in rows])

    ax = fig.add_subplot(111)

    # Colour map: absent = light grey, present = row colour
    # We use a simple approach: draw a grey background then overlay colored cells
    ax.imshow(
        np.isnan(matrix).astype(float),
        aspect='auto',
        cmap='gray_r',
        vmin=0, vmax=2,
        extent=[t_ds[0], t_ds[-1], -0.5, len(rows) - 0.5],
        origin='lower',
        interpolation='nearest',
    )
    # Overlay presence per satellite in constellation colour
    for k, (label, arr_full, color) in enumerate(rows):
        pres = np.isfinite(arr_full[::step]).astype(float)
        pres[pres == 0] = np.nan
        pres_2d = pres[np.newaxis, :]  # (1, T)
        # Create single-color image
        rgb = mcolors.to_rgb(color)
        img = np.ones((1, len(t_ds), 4))
        img[0, :, 0] = rgb[0]
        img[0, :, 1] = rgb[1]
        img[0, :, 2] = rgb[2]
        img[0, :, 3] = np.where(np.isfinite(pres), 0.85, 0.0)
        ax.imshow(
            img,
            aspect='auto',
            extent=[t_ds[0], t_ds[-1], k - 0.5, k + 0.5],
            origin='lower',
            interpolation='nearest',
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(t_ds[0], t_ds[-1])
    ax.set_xlabel(_xlabel(times, file_name), fontsize=8)
    ax.set_title('Observable Availability Map  (coloured = present, grey = gap)')
    ax.grid(True, axis='x', alpha=0.2)
    fig.tight_layout()


# ===========================================================================
# 3. C/N0 over time
# ===========================================================================

def plot_rinex_cn0(rinex_obs, file_name=''):
    """Plot signal strength (C/N0) over time, one line per satellite.

    Uses ``S*`` observables (direct C/N0 in dB-Hz) when available, falling
    back to the SNR flag embedded in ``L*`` observations.

    Parameters
    ----------
    rinex_obs : dict
        As returned by :func:`read_rinex_obs`.
    file_name : str, optional
        Source file name.
    """
    times = rinex_obs['times']
    sats  = rinex_obs['sats']

    if not sats or len(times) == 0:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=plt.gca().transAxes)
        return

    fig = plt.gcf()
    fig.clf()
    ax  = fig.add_subplot(111)

    _order = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4, 'S': 5, 'I': 6, '?': 7}
    sorted_sats = sorted(sats, key=lambda s: (_order.get(s[0], 9), s))

    any_data = False
    for sv in sorted_sats:
        code, cn0 = get_signal_strength(rinex_obs, sv)
        if cn0 is None or not np.any(np.isfinite(cn0)):
            continue
        any_data = True
        color = _color(sv)
        (h,) = ax.plot(times, cn0, lw=0.8, alpha=0.85, color=color, label=sv)
        # Annotate at the last valid point
        i_f = np.where(np.isfinite(cn0))[0]
        if len(i_f) > 0:
            ax.text(times[i_f[-1]], cn0[i_f[-1]], sv,
                    color=color, fontsize=7, va='center')

    if not any_data:
        ax.text(0.5, 0.5,
                'No signal-strength observables (S* or SNR flag) found',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('C/N0 over Time')
        return

    # Legend per constellation
    seen_sys = {s[0] for s in sorted_sats}
    patches = [
        mpatches.Patch(color=_SYS_COLOR.get(sc, '#888'),
                       label=_SYS_NAME.get(sc, sc))
        for sc in sorted('GREJCSI', key=lambda x: _order.get(x, 9))
        if sc in seen_sys
    ]
    if patches:
        ax.legend(handles=patches, fontsize=8, loc='lower right',
                  framealpha=0.8)

    ax.set_xlabel(_xlabel(times, file_name), fontsize=8)
    ax.set_ylabel('Signal strength (dB-Hz)')
    ax.set_title('C/N0 over Time')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()


# ===========================================================================
# 4. Cycle-slip timeline
# ===========================================================================

def plot_rinex_cycle_slips(rinex_obs, file_name=''):
    """Plot cycle-slip detections from the LLI flag.

    The Loss-of-Lock Indicator (LLI) bit 0 signals a cycle slip.  This plot
    shows a timeline of detected slips per satellite, with a count summary.

    Parameters
    ----------
    rinex_obs : dict
        As returned by :func:`read_rinex_obs`.
    file_name : str, optional
        Source file name.
    """
    times = rinex_obs['times']
    sats  = rinex_obs['sats']
    lli   = rinex_obs['lli']

    if not sats or len(times) == 0:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=plt.gca().transAxes)
        return

    fig = plt.gcf()
    fig.clf()

    _order = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4, 'S': 5, 'I': 6, '?': 7}
    sorted_sats = sorted(sats, key=lambda s: (_order.get(s[0], 9), s))

    # Compute slip mask per satellite (OR across all phase observables)
    slip_times  : dict[str, np.ndarray] = {}
    slip_counts : dict[str, int]         = {}

    for sv in sorted_sats:
        if sv not in lli:
            continue
        # Aggregate over all L* observables
        slip_mask = np.zeros(len(times), dtype=bool)
        for code, lli_arr in lli[sv].items():
            if not code.startswith('L'):
                continue
            slip_mask |= (lli_arr.astype(int) & 0x01).astype(bool)
        slip_times[sv]  = times[slip_mask]
        slip_counts[sv] = int(np.sum(slip_mask))

    # Filter to satellites that have phase observables
    active_sats = [sv for sv in sorted_sats if sv in slip_times]

    if not active_sats:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No phase (L*) observables found in LLI data',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Cycle Slips from LLI Flag')
        return

    # Two sub-panels: (left) timeline, (right) bar chart of counts
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_time  = fig.add_subplot(gs[0])
    ax_count = fig.add_subplot(gs[1], sharey=ax_time)

    total_slips = sum(slip_counts.values())

    for k, sv in enumerate(active_sats):
        color = _color(sv)
        y = len(active_sats) - 1 - k  # top satellite first
        t_slips = slip_times[sv]
        if len(t_slips) > 0:
            ax_time.scatter(t_slips, [y] * len(t_slips),
                            marker='|', s=60, color=color, linewidths=1.5)
        # Horizontal bar (tracking interval)
        ax_time.barh(y, times[-1] - times[0], left=times[0],
                     height=0.4, color=color, alpha=0.15, linewidth=0)

    # Y-axis labels
    ax_time.set_yticks(range(len(active_sats)))
    ax_time.set_yticklabels(list(reversed(active_sats)), fontsize=8)
    ax_time.set_ylim(-0.5, len(active_sats) - 0.5)
    ax_time.set_xlim(times[0], times[-1])
    ax_time.set_xlabel(_xlabel(times, file_name), fontsize=8)
    ax_time.set_title(
        f'Cycle Slips from LLI Flag  (total {total_slips} slips)'
    )
    ax_time.grid(True, axis='x', alpha=0.3)

    # Bar chart of slip counts
    ys      = list(range(len(active_sats)))
    counts  = [slip_counts[sv] for sv in reversed(active_sats)]
    colors  = [_color(sv) for sv in reversed(active_sats)]
    ax_count.barh(ys, counts, color=colors, alpha=0.75)
    ax_count.set_xlabel('# slips', fontsize=8)
    ax_count.yaxis.set_visible(False)
    ax_count.grid(True, axis='x', alpha=0.3)
    ax_count.set_xlim(left=0)

    # Constellation legend
    seen_sys = {s[0] for s in active_sats}
    patches = [
        mpatches.Patch(color=_SYS_COLOR.get(sc, '#888'),
                       label=_SYS_NAME.get(sc, sc))
        for sc in sorted('GREJCSI', key=lambda x: _order.get(x, 9))
        if sc in seen_sys
    ]
    if patches:
        ax_time.legend(handles=patches, fontsize=8, loc='lower right',
                       framealpha=0.8)

    fig.tight_layout()
