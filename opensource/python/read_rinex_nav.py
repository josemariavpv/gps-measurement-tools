"""Read RINEX 2.10 navigation (ephemeris) files.

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

from .utc2gps import utc2gps


def _str2num(s):
    """Convert RINEX field string to float, handling 'D' exponent notation."""
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


def _count_eph(lines):
    """Find the line index after the header and count ephemeris records."""
    header_end = None
    for i, line in enumerate(lines):
        if 'END OF HEADER' in line:
            header_end = i
            break
    if header_end is None:
        raise ValueError('Expected RINEX header not found')
    data_lines = lines[header_end + 1:]
    # Remove trailing empty lines
    while data_lines and data_lines[-1].strip() == '':
        data_lines.pop()
    if len(data_lines) % 8 != 0:
        raise ValueError(
            f'Number of nav lines ({len(data_lines)}) should be divisible by 8')
    return header_end, len(data_lines) // 8


def _read_iono(lines, num_hdr_lines):
    """Read ionospheric parameters from RINEX header."""
    iono = None
    alpha = None
    beta = None
    for line in lines[:num_hdr_lines]:
        if 'ION ALPHA' in line:
            idx = line.index('ION ALPHA')
            parts = line[:idx].split()
            if len(parts) >= 4:
                alpha = [float(p.replace('D', 'E').replace('d', 'e'))
                         for p in parts[:4]]
        if 'ION BETA' in line:
            idx = line.index('ION BETA')
            parts = line[:idx].split()
            if len(parts) >= 4:
                beta = [float(p.replace('D', 'E').replace('d', 'e'))
                        for p in parts[:4]]
    if alpha is not None and beta is not None:
        iono = {'alpha': alpha, 'beta': beta}
    return iono


def read_rinex_nav(file_name):
    """Read GPS ephemeris and iono data from an ASCII RINEX 2.10 Nav file.

    Parameters
    ----------
    file_name : str
        Path to the RINEX navigation data file.

    Returns
    -------
    gps_eph : list of dict
        Each dict contains the ephemeris fields for one satellite epoch.
        Field names follow RINEX 2.1 Table A4 / IS-GPS-200.
    iono : dict or None
        Ionospheric parameters {'alpha': [...], 'beta': [...]}, or None.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()

    header_end, num_eph = _count_eph(list(lines))
    iono = _read_iono(lines, header_end)

    gps_eph = []
    idx = header_end + 1  # start of data lines

    for _ in range(num_eph):
        eph = _init_eph()
        line = lines[idx]; idx += 1
        # Pad line to at least 79 characters
        line = line.rstrip('\n').ljust(79)

        eph['PRN'] = int(line[0:2])
        year = int(line[2:6])
        if year < 80:
            year = 2000 + year
        else:
            year = 1900 + year
        month  = int(line[6:9])
        day    = int(line[9:12])
        hour   = int(line[12:15])
        minute = int(line[15:18])
        second = float(line[18:22])

        gps_time, _ = utc2gps([year, month, day, hour, minute, second])
        eph['Toc'] = float(gps_time[0, 1])

        eph['af0'] = _str2num(line[22:41])
        eph['af1'] = _str2num(line[41:60])
        eph['af2'] = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['IODE']    = _str2num(line[3:22])
        eph['Crs']     = _str2num(line[22:41])
        eph['Delta_n'] = _str2num(line[41:60])
        eph['M0']      = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['Cuc']    = _str2num(line[3:22])
        eph['e']      = _str2num(line[22:41])
        eph['Cus']    = _str2num(line[41:60])
        eph['Asqrt']  = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['Toe']   = _str2num(line[3:22])
        eph['Cic']   = _str2num(line[22:41])
        eph['OMEGA'] = _str2num(line[41:60])
        eph['Cis']   = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['i0']        = _str2num(line[3:22])
        eph['Crc']       = _str2num(line[22:41])
        eph['omega']     = _str2num(line[41:60])
        eph['OMEGA_DOT'] = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['IDOT']     = _str2num(line[3:22])
        eph['codeL2']   = _str2num(line[22:41])
        eph['GPS_Week'] = _str2num(line[41:60])
        eph['L2Pdata']  = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['accuracy'] = _str2num(line[3:22])
        eph['health']   = _str2num(line[22:41])
        eph['TGD']      = _str2num(line[41:60])
        eph['IODC']     = _str2num(line[60:79])

        line = lines[idx].rstrip('\n').ljust(79); idx += 1
        eph['ttx']          = _str2num(line[3:22])
        eph['Fit_interval'] = _str2num(line[22:41])

        gps_eph.append(eph)

    return gps_eph, iono
