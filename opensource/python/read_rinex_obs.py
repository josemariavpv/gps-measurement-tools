"""Read RINEX 2.x / 3.x observation files (ReadRinexObs).

Parses the observation records into a structured dict suitable for quality
analysis: satellite visibility, C/N0, observable availability and cycle-slip
detection from the LLI flag.

Supported formats
-----------------
* RINEX 2.10, 2.11  (single-constellation GPS and multi-GNSS mixed)
* RINEX 3.02, 3.03, 3.04  (mixed multi-GNSS)

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

import datetime
import math
import re
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_rinex_obs(file_name):
    """Read a RINEX 2 or 3 observation file.

    Parameters
    ----------
    file_name : str
        Path to the RINEX observation file (.obs, .rnx, .YYo, etc.).

    Returns
    -------
    rinex_obs : dict or None
        None if the file cannot be parsed.  Otherwise a dict with:

        'version'    – float, RINEX version (e.g. 3.03)
        'times'      – ndarray (N,), seconds since first epoch
        'datetimes'  – list of N datetime.datetime objects (UTC)
        'sats'       – sorted list of M satellite IDs  (e.g. 'G01', 'E02')
        'obs_types'  – dict  sys_char → list of obs-type codes
                       sys_char ∈ {'G','R','E','C','J','S','I','?'}
        'data'       – dict  sat_id → dict  obs_code → ndarray (N,) float
        'lli'        – dict  sat_id → dict  obs_code → ndarray (N,) uint8
        'snr_flag'   – dict  sat_id → dict  obs_code → ndarray (N,) uint8
    """
    try:
        with open(file_name, 'r', errors='replace') as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f'read_rinex_obs: cannot open {file_name}: {exc}')
        return None

    # ---- Header --------------------------------------------------------
    header_end, version, obs_types_hdr = _parse_header(lines)
    if header_end is None:
        print('read_rinex_obs: END OF HEADER not found')
        return None

    # ---- Data ----------------------------------------------------------
    data_lines = lines[header_end + 1:]

    if version >= 3.0:
        epochs, epoch_sats, epoch_obs = _parse_v3(data_lines, obs_types_hdr)
    else:
        epochs, epoch_sats, epoch_obs = _parse_v2(data_lines, obs_types_hdr)

    if not epochs:
        print('read_rinex_obs: no epochs found')
        return None

    # ---- Collect all satellite IDs and build output arrays -------------
    all_sats = sorted({s for sats in epoch_sats for s in sats})
    n = len(epochs)
    m = len(all_sats)

    # obs_types per system (from what was actually observed)
    obs_types = {}
    for sc in {s[0] for s in all_sats}:
        obs_types[sc] = obs_types_hdr.get(sc, [])

    # For each sat×obs_code build arrays over time
    # First pass: collect all obs codes for each sat
    sat_obs_codes: dict[str, set] = {s: set() for s in all_sats}
    for i_e, sats in enumerate(epoch_sats):
        for sv in sats:
            if sv in epoch_obs[i_e]:
                sat_obs_codes[sv].update(epoch_obs[i_e][sv].keys())

    # Build ndarrays (NaN = missing)
    data    : dict[str, dict[str, np.ndarray]] = {}
    lli     : dict[str, dict[str, np.ndarray]] = {}
    snr_flag: dict[str, dict[str, np.ndarray]] = {}

    for sv in all_sats:
        codes = sorted(sat_obs_codes[sv])
        data[sv]     = {c: np.full(n, np.nan) for c in codes}
        lli[sv]      = {c: np.zeros(n, dtype=np.uint8) for c in codes}
        snr_flag[sv] = {c: np.zeros(n, dtype=np.uint8) for c in codes}

    for i_e, sats in enumerate(epoch_sats):
        obs_epoch = epoch_obs[i_e]
        for sv in sats:
            if sv not in obs_epoch:
                continue
            for code, (val, lli_v, snr_v) in obs_epoch[sv].items():
                if code in data[sv]:
                    data[sv][code][i_e]     = val
                    lli[sv][code][i_e]      = lli_v
                    snr_flag[sv][code][i_e] = snr_v

    t0 = epochs[0]
    times = np.array([(ep - t0).total_seconds() for ep in epochs])

    return {
        'version':   version,
        'times':     times,
        'datetimes': epochs,
        'sats':      all_sats,
        'obs_types': obs_types,
        'data':      data,
        'lli':       lli,
        'snr_flag':  snr_flag,
    }


# ---------------------------------------------------------------------------
# Signal-strength helper
# ---------------------------------------------------------------------------

def get_signal_strength(rinex_obs, sat_id):
    """Return the best signal-strength (C/N0 proxy) array for a satellite.

    Looks for ``S*`` observables first; falls back to the SNR flag embedded
    in the LLI/SNR nibble of ``L*`` observables (scaled to dB-Hz approximation
    using the RINEX convention SNR_flag × 6 + 6).

    Parameters
    ----------
    rinex_obs : dict
        As returned by :func:`read_rinex_obs`.
    sat_id : str
        E.g. ``'G01'``.

    Returns
    -------
    tuple (obs_code, ndarray) or (None, None)
    """
    if sat_id not in rinex_obs['data']:
        return None, None
    codes = rinex_obs['data'][sat_id]
    # Prefer S1C, S1P, S1X, S1W, S2C, S5Q …
    s_codes = sorted([c for c in codes if c.startswith('S')],
                     key=lambda c: (c[1], c))  # sort by freq band, then code
    for c in s_codes:
        arr = codes[c]
        if np.any(np.isfinite(arr)):
            return c, arr
    # Fallback: use embedded SNR flag from L1 obs
    l_codes = sorted([c for c in codes if c.startswith('L')],
                     key=lambda c: (c[1], c))
    for c in l_codes:
        snr_arr = rinex_obs['snr_flag'][sat_id][c].astype(float)
        snr_arr[snr_arr == 0] = np.nan
        snr_dbhz = snr_arr * 6 + 6  # RINEX SNR flag → approximate dB-Hz
        if np.any(np.isfinite(snr_dbhz)):
            return c + '_snr', snr_dbhz
    return None, None


# ---------------------------------------------------------------------------
# Private – header parsing
# ---------------------------------------------------------------------------

def _parse_header(lines):
    """Return (header_end_index, version, obs_types_dict)."""
    version     = 2.10
    obs_types   = {}   # sys_char -> [obs_code, ...]
    header_end  = None
    # RINEX 2 accumulator
    v2_types    = []

    for i, line in enumerate(lines):
        # Search the whole line for known RINEX header labels.
        # The RINEX standard places labels in cols 61-80, but many tools
        # use slightly different padding, so we check the full line.

        if 'RINEX VERSION / TYPE' in line:
            try:
                version = float(line[:9].strip())
            except ValueError:
                pass

        elif 'SYS / # / OBS TYPES' in line:
            # RINEX 3 – may span multiple continuation lines
            sys_char = line[0].upper()
            if sys_char == ' ':
                # continuation line for the same system
                last_sys = list(obs_types.keys())[-1] if obs_types else 'G'
                sys_char = last_sys
            try:
                n_types = int(line[3:6].strip() or 0)
            except ValueError:
                n_types = 0
            codes = _read_obs_codes(line[7:60])
            obs_types.setdefault(sys_char, []).extend(codes)
            # Read continuation lines (each holds up to 13 codes)
            j = i + 1
            while len(obs_types[sys_char]) < n_types and j < len(lines):
                cont = lines[j]
                if 'SYS / # / OBS TYPES' not in cont:
                    # continuation line (label field is blank)
                    obs_types[sys_char].extend(_read_obs_codes(cont[7:60]))
                    j += 1
                else:
                    break

        elif '# / TYPES OF OBSERV' in line:
            # RINEX 2 – one line (+ possible continuation)
            try:
                n_types = int(line[:6].strip())
            except ValueError:
                n_types = 0
            codes = [line[10 + k * 6: 12 + k * 6].strip()
                     for k in range(min(9, n_types)) if line[10 + k * 6: 12 + k * 6].strip()]
            v2_types.extend(codes)

        elif 'END OF HEADER' in line:
            header_end = i
            break

    if version < 3.0 and v2_types:
        # RINEX 2: all constellations share the same obs types
        obs_types['?'] = v2_types

    return header_end, version, obs_types


def _read_obs_codes(s):
    """Parse up to 13 obs-type codes from a 53-char RINEX 3 field."""
    codes = []
    for k in range(13):
        start = k * 4
        end   = start + 4
        code  = s[start:end].strip() if len(s) >= end else ''
        if code:
            codes.append(code)
    return codes


# ---------------------------------------------------------------------------
# Private – RINEX 3 data parsing
# ---------------------------------------------------------------------------

def _parse_v3(lines, obs_types_hdr):
    """Parse RINEX 3 data records.

    Returns (epochs, epoch_sats, epoch_obs):
        epochs     – list of datetime
        epoch_sats – list[list[str]]   satellite IDs per epoch
        epoch_obs  – list[dict]        {sat_id: {obs_code: (val,lli,snr)}}
    """
    epochs     = []
    epoch_sats = []
    epoch_obs  = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith('>'):
            # Epoch header
            ep_dt = _parse_epoch_v3(line)
            if ep_dt is None:
                i += 1
                continue
            try:
                num_sats = int(line[32:35].strip())
            except (ValueError, IndexError):
                num_sats = 0

            sats_epoch = []
            obs_epoch  = {}
            i += 1
            for _ in range(num_sats):
                if i >= n:
                    break
                sat_line = lines[i]
                i += 1
                if len(sat_line) < 3:
                    continue
                sat_id = sat_line[:3].strip()
                if not sat_id or len(sat_id) < 3:
                    continue
                # Normalise sat_id: e.g. 'G 1' → 'G01'
                sys_char = sat_id[0].upper()
                try:
                    prn = int(sat_id[1:3])
                except ValueError:
                    continue
                sat_id = f'{sys_char}{prn:02d}'
                sats_epoch.append(sat_id)

                obs_codes = obs_types_hdr.get(sys_char, obs_types_hdr.get('?', []))
                obs_epoch[sat_id] = _parse_obs_line_v3(sat_line[3:], obs_codes)

            epochs.append(ep_dt)
            epoch_sats.append(sats_epoch)
            epoch_obs.append(obs_epoch)
        else:
            i += 1

    return epochs, epoch_sats, epoch_obs


def _parse_epoch_v3(line):
    """Parse "> YYYY MM DD hh mm ss.sssssss  flag  numSV" line."""
    try:
        yr  = int(line[2:6])
        mo  = int(line[7:9])
        dy  = int(line[10:12])
        hr  = int(line[13:15])
        mi  = int(line[16:18])
        sec = float(line[19:29])
        isec = int(sec)
        usec = int(round((sec - isec) * 1e6))
        return datetime.datetime(yr, mo, dy, hr, mi, isec, usec)
    except (ValueError, IndexError):
        return None


def _parse_obs_line_v3(s, obs_codes):
    """Parse one satellite observation line (chars after the 3-char sat ID).

    Returns dict  obs_code → (value, lli_flag, snr_flag).
    """
    result = {}
    for k, code in enumerate(obs_codes):
        start = k * 16
        end   = start + 16
        field = s[start:end] if len(s) >= end else ''
        field = field.ljust(16)
        val_s = field[:14].strip()
        try:
            val = float(val_s) if val_s else np.nan
        except ValueError:
            val = np.nan
        try:
            lli_v = int(field[14]) if field[14].strip() else 0
        except (ValueError, IndexError):
            lli_v = 0
        try:
            snr_v = int(field[15]) if field[15].strip() else 0
        except (ValueError, IndexError):
            snr_v = 0
        result[code] = (val, lli_v, snr_v)
    return result


# ---------------------------------------------------------------------------
# Private – RINEX 2 data parsing
# ---------------------------------------------------------------------------

_V2_SAT_RE = re.compile(r'([GREJCSI])(\d{2})')


def _parse_v2(lines, obs_types_hdr):
    """Parse RINEX 2.x data records."""
    obs_codes_all = obs_types_hdr.get('?', [])  # shared for all constellations
    n_obs = len(obs_codes_all)

    epochs     = []
    epoch_sats = []
    epoch_obs  = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Epoch header: ' YY MM DD hh mm ss.sssssss  0 numSV  sat...'
        if len(line) >= 26 and line[0] == ' ' and _looks_like_v2_epoch(line):
            ep_dt = _parse_epoch_v2(line)
            if ep_dt is None:
                i += 1
                continue
            try:
                num_sats = int(line[29:32].strip())
            except (ValueError, IndexError):
                i += 1
                continue

            # Satellite list (up to 12 per line, continuation if >12)
            sat_ids = []
            lines_needed = math.ceil(num_sats / 12)
            for k_line in range(lines_needed):
                if i + k_line >= n:
                    break
                sl = lines[i + k_line]
                col_start = 32 if k_line == 0 else 32
                if k_line > 0:
                    col_start = 32  # continuation: same format but col 32..68
                    sl = ' ' * 32 + sl[:36]
                for k_sv in range(12):
                    col = 32 + k_sv * 3
                    sv_s = sl[col:col + 3] if len(sl) >= col + 3 else ''
                    sv_s = sv_s.strip()
                    if not sv_s:
                        continue
                    # GPS default if no sys char
                    if sv_s[0].isdigit():
                        sv_s = 'G' + sv_s.zfill(2)
                    elif len(sv_s) == 2 and sv_s[0].upper() in 'GREJCSI':
                        sv_s = sv_s[0].upper() + sv_s[1:].zfill(2)
                    try:
                        sys_c = sv_s[0].upper()
                        prn   = int(sv_s[1:])
                        sat_ids.append(f'{sys_c}{prn:02d}')
                    except (ValueError, IndexError):
                        pass

            i += lines_needed  # skip epoch + continuation header lines

            obs_epoch = {}
            # Read observation lines for each satellite
            obs_per_line = 5
            lines_per_sat = math.ceil(n_obs / obs_per_line) if n_obs > 0 else 1

            for sv in sat_ids:
                obs_sv = {}
                all_vals = []
                all_lli  = []
                all_snr  = []
                for k_line in range(lines_per_sat):
                    if i >= n:
                        break
                    ol = lines[i]
                    i += 1
                    for k_obs in range(obs_per_line):
                        start = k_obs * 16
                        end   = start + 16
                        field = ol[start:end] if len(ol) >= end else ''
                        field = field.ljust(16)
                        val_s = field[:14].strip()
                        try:
                            val = float(val_s) if val_s else np.nan
                        except ValueError:
                            val = np.nan
                        try:
                            lli_v = int(field[14]) if field[14].strip() else 0
                        except (ValueError, IndexError):
                            lli_v = 0
                        try:
                            snr_v = int(field[15]) if field[15].strip() else 0
                        except (ValueError, IndexError):
                            snr_v = 0
                        all_vals.append(val)
                        all_lli.append(lli_v)
                        all_snr.append(snr_v)

                for k_obs, code in enumerate(obs_codes_all):
                    if k_obs < len(all_vals):
                        obs_sv[code] = (all_vals[k_obs],
                                        all_lli[k_obs],
                                        all_snr[k_obs])
                    else:
                        obs_sv[code] = (np.nan, 0, 0)
                obs_epoch[sv] = obs_sv

            epochs.append(ep_dt)
            epoch_sats.append(sat_ids)
            epoch_obs.append(obs_epoch)
        else:
            i += 1

    return epochs, epoch_sats, epoch_obs


def _looks_like_v2_epoch(line):
    """Heuristic check: is this line a RINEX 2 epoch header?"""
    try:
        int(line[1:3])   # YY
        int(line[4:6])   # MM
        int(line[7:9])   # DD
        int(line[10:12]) # hh
        int(line[13:15]) # mm
        float(line[15:26])  # ss.sssssss
        return True
    except (ValueError, IndexError):
        return False


def _parse_epoch_v2(line):
    """Parse RINEX 2 epoch record first line."""
    try:
        yr  = int(line[1:3])
        mo  = int(line[4:6])
        dy  = int(line[7:9])
        hr  = int(line[10:12])
        mi  = int(line[13:15])
        sec = float(line[15:26])
        yr  = 2000 + yr if yr < 80 else 1900 + yr
        isec = int(sec)
        usec = int(round((sec - isec) * 1e6))
        return datetime.datetime(yr, mo, dy, hr, mi, isec, usec)
    except (ValueError, IndexError):
        return None
