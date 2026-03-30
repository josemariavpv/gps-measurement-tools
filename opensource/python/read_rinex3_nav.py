"""Read RINEX 3.x mixed navigation files (GPS and Galileo).

Parses RINEX 3 broadcast navigation files and returns GPS and Galileo
ephemeris records in the same dict format as :func:`read_rinex_nav` so that
the existing :func:`gps_eph2xyz` function works for both constellations.

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

import gzip
import os

from .utc2gps import utc2gps


def _str2num(s):
    """Convert a RINEX field string to float, handling 'D' exponent notation."""
    s = s.strip()
    if not s:
        return 0.0
    s = s.replace('D', 'E').replace('d', 'e')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _init_eph():
    """Return a dict with all ephemeris fields initialised to zero."""
    return {
        'PRN': 0,
        'Toc': 0.0,
        'af0': 0.0,
        'af1': 0.0,
        'af2': 0.0,
        'IODE': 0.0,
        'Crs': 0.0,
        'Delta_n': 0.0,
        'M0': 0.0,
        'Cuc': 0.0,
        'e': 0.0,
        'Cus': 0.0,
        'Asqrt': 0.0,
        'Toe': 0.0,
        'Cic': 0.0,
        'OMEGA': 0.0,
        'Cis': 0.0,
        'i0': 0.0,
        'Crc': 0.0,
        'omega': 0.0,
        'OMEGA_DOT': 0.0,
        'IDOT': 0.0,
        'codeL2': 0.0,
        'GPS_Week': 0.0,
        'L2Pdata': 0.0,
        'accuracy': 0.0,
        'health': 0.0,
        'TGD': 0.0,
        'IODC': 0.0,
        'ttx': 0.0,
        'Fit_interval': 0.0,
    }


def _parse_epoch_line(line):
    """Parse the epoch line of a RINEX 3 navigation record.

    Returns (constellation_char, prn, year, month, day, hour, min, sec,
             af0, af1, af2) or None if the line is not a nav epoch line.
    """
    if len(line) < 4:
        return None
    const = line[0]
    if const not in ('G', 'E'):
        return None
    try:
        prn = int(line[1:3])
    except ValueError:
        return None
    # Epoch: YYYY MM DD HH MM SS
    # Columns 4..22 = year(4) + 2-char fields separated by space
    try:
        year   = int(line[4:8])
        month  = int(line[9:11])
        day    = int(line[12:14])
        hour   = int(line[15:17])
        minute = int(line[18:20])
        second = float(line[21:23])
    except (ValueError, IndexError):
        return None

    # af0, af1, af2 at columns 23..60 (3 × 19 chars)
    rest = line[23:].ljust(57)
    af0 = _str2num(rest[0:19])
    af1 = _str2num(rest[19:38])
    af2 = _str2num(rest[38:57])

    return const, prn, year, month, day, hour, minute, second, af0, af1, af2


def _parse_cont(line):
    """Parse a RINEX 3 continuation line (4 leading spaces, 4 × 19 chars)."""
    line = line.rstrip('\n').ljust(80)
    # RINEX 3 continuation lines have 4 leading spaces
    data = line[4:].ljust(76)
    v0 = _str2num(data[0:19])
    v1 = _str2num(data[19:38])
    v2 = _str2num(data[38:57])
    v3 = _str2num(data[57:76])
    return v0, v1, v2, v3


def read_rinex3_nav(file_name):
    """Read GPS and Galileo ephemeris from a RINEX 3 mixed navigation file.

    Handles both plain text (``.rnx``) and gzip-compressed (``.rnx.gz``)
    files.

    Parameters
    ----------
    file_name : str
        Path to the RINEX 3 navigation file (optionally gzip-compressed).

    Returns
    -------
    gps_eph : list of dict
        GPS ephemeris records in the same format as :func:`read_rinex_nav`.
    gal_eph : list of dict
        Galileo ephemeris records in the same format (``GPS_Week`` holds the
        Galileo system week; orbital parameters are identical to GPS).
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f'Navigation file not found: {file_name}')

    # Support gzip-compressed files transparently
    opener = gzip.open if file_name.endswith('.gz') else open
    with opener(file_name, 'rt', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()

    # Find end of header
    header_end = None
    for i, line in enumerate(lines):
        if 'END OF HEADER' in line:
            header_end = i
            break
    if header_end is None:
        raise ValueError(f'RINEX header end not found in: {file_name}')

    gps_eph = []
    gal_eph = []

    i = header_end + 1
    n = len(lines)

    while i < n:
        line = lines[i]
        parsed = _parse_epoch_line(line)
        if parsed is None:
            i += 1
            continue

        const, prn, year, month, day, hour, minute, second, af0, af1, af2 = parsed
        i += 1

        eph = _init_eph()
        eph['PRN'] = prn
        eph['af0'] = af0
        eph['af1'] = af1
        eph['af2'] = af2

        gps_time_arr, _ = utc2gps([year, month, day, hour, minute, second])
        eph['Toc'] = float(gps_time_arr[0, 1])

        # Read 7 continuation lines (same layout for GPS and Galileo)
        if i + 7 > n:
            break

        # Line 1: IODE, Crs, Delta_n, M0
        v = _parse_cont(lines[i]); i += 1
        eph['IODE'], eph['Crs'], eph['Delta_n'], eph['M0'] = v

        # Line 2: Cuc, e, Cus, sqrt(A)
        v = _parse_cont(lines[i]); i += 1
        eph['Cuc'], eph['e'], eph['Cus'], eph['Asqrt'] = v

        # Line 3: Toe, Cic, OMEGA, Cis
        v = _parse_cont(lines[i]); i += 1
        eph['Toe'], eph['Cic'], eph['OMEGA'], eph['Cis'] = v

        # Line 4: i0, Crc, omega, OMEGA_DOT
        v = _parse_cont(lines[i]); i += 1
        eph['i0'], eph['Crc'], eph['omega'], eph['OMEGA_DOT'] = v

        # Line 5: IDOT, (Data_sources / codeL2), week, (SISA / L2P)
        v = _parse_cont(lines[i]); i += 1
        eph['IDOT']     = v[0]
        eph['codeL2']   = v[1]
        eph['GPS_Week'] = v[2]   # GAL week for Galileo, GPS week for GPS
        eph['L2Pdata']  = v[3]

        # Line 6 layout differs between GPS and Galileo (RINEX 3.03 spec):
        #   GPS:     v[0]=SV_accuracy, v[1]=SV_health, v[2]=TGD,        v[3]=IODC
        #   Galileo: v[0]=SV_health,   v[1]=BGD_E5a/E1, v[2]=BGD_E5b/E1, v[3]=spare
        v = _parse_cont(lines[i]); i += 1
        if const == 'G':
            eph['accuracy'] = v[0]
            eph['health']   = v[1]
            eph['TGD']      = v[2]
            eph['IODC']     = v[3]
        else:  # Galileo
            eph['health']   = v[0]
            eph['TGD']      = v[1]   # BGD_E5a/E1

        # Line 7: transmission time, fit interval, ...
        v = _parse_cont(lines[i]); i += 1
        eph['ttx']          = v[0]
        # GPS fit interval is in line 7; for Galileo ICD the validity period
        # is typically 4 hours (half of the standard 8-hour slot), so 4.0 h
        # is the appropriate fallback when no explicit value is provided.
        eph['Fit_interval'] = v[1] if const == 'G' else 4.0

        if const == 'G':
            gps_eph.append(eph)
        elif const == 'E':
            gal_eph.append(eph)

    return gps_eph, gal_eph
