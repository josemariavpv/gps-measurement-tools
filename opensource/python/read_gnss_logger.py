"""Read GnssLogger log files from Android (ReadGnssLogger).

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
import csv
import numpy as np

from .check_data_filter import check_data_filter
from .compare_versions import compare_versions


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------

def read_gnss_logger(dir_name, file_name, data_filter=None, gnss_analysis=None):
    """Read a GnssLogger log file produced by the Android GnssLogger app.

    Compatible with Android release N (log version ≥ 1.4.0.0).

    Parameters
    ----------
    dir_name : str
        Directory containing the log file.
    file_name : str
        Log file name (``*.txt`` or ``*.csv``).
    data_filter : list of (str, str), optional
        Filter conditions; see set_data_filter() for format.
    gnss_analysis : dict, optional
        Existing analysis dict that will be updated.

    Returns
    -------
    gnss_raw : dict or None
        All GnssClock and GnssMeasurement fields plus
        ``gnss_raw['allRxMillis']`` (int64 array, full-cycle time in ms).
    gnss_analysis : dict
        Analysis / pass-fail information.
    """
    if gnss_analysis is None:
        gnss_analysis = {}
    gnss_analysis['GnssClockErrors']       = 'GnssClock Errors.'
    gnss_analysis['GnssMeasurementErrors'] = 'GnssMeasurement Errors.'
    gnss_analysis['ApiPassFail']           = ''

    ext = file_name[-4:].lower()
    if ext not in ('.txt', '.csv'):
        raise ValueError(
            'Expecting file name of the form "*.txt" or "*.csv"')

    if not dir_name.endswith(os.sep):
        dir_name = dir_name + os.sep

    # ------------------------------------------------------------------
    # 1. Convert .txt → csv (or use .csv directly)
    # ------------------------------------------------------------------
    raw_csv = _make_csv(dir_name, file_name)

    # ------------------------------------------------------------------
    # 2. Read the CSV
    # ------------------------------------------------------------------
    header, C = _read_raw_csv(raw_csv)

    # ------------------------------------------------------------------
    # 3. Apply data filter
    # ------------------------------------------------------------------
    ok = check_data_filter(data_filter, header)
    if not ok:
        return None, gnss_analysis

    ok, C = _filter_data(C, data_filter, header)
    if not ok:
        return None, gnss_analysis

    # ------------------------------------------------------------------
    # 4. Pack into gnss_raw dict
    # ------------------------------------------------------------------
    gnss_raw, missing = _pack_gnss_raw(C, header)

    # ------------------------------------------------------------------
    # 5. Check clock and report missing fields
    # ------------------------------------------------------------------
    gnss_raw, gnss_analysis = _check_gnss_clock(gnss_raw, gnss_analysis)
    gnss_analysis = _report_missing_fields(gnss_analysis, missing)

    return gnss_raw, gnss_analysis


# -------------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------------

_INT64_FIELDS = {
    'TimeNanos', 'FullBiasNanos',
    'ReceivedSvTimeNanos', 'ReceivedSvTimeUncertaintyNanos', 'CarrierCycles',
}
_STRING_FIELDS = {'CodeType'}


def _make_csv(dir_name, file_name):
    """Convert .txt GnssLogger file to a .csv file and return path."""
    csv_path = os.path.join(dir_name, 'raw.csv')
    full_path = os.path.join(dir_name, file_name)

    if file_name.lower().endswith('.csv'):
        return full_path  # nothing to do

    print(f'\nReading file {full_path}')

    # --- read version line ------------------------------------------------
    version = None
    with open(full_path, 'r', errors='replace') as f:
        for line in f:
            if 'version' in line.lower():
                import re
                digits = re.search(r'(\d+)\.(\d+)\.(\d+)\.(\d+)', line)
                if digits:
                    version = tuple(int(x) for x in digits.groups())
                else:
                    m = re.search(r'\d+', line)
                    if m:
                        start = m.start()
                        parts = re.findall(r'\d+', line[start:])
                        version = tuple(int(p) for p in parts[:4])
                        while len(version) < 4:
                            version = version + (0,)
                break

    if version is None:
        print(f'Could not find "Version" in input file {file_name}')
        version = (1, 4, 0, 0)

    min_version = (1, 4, 0, 0)
    if compare_versions(version, min_version) == 'before':
        raise ValueError(
            f'This version of read_gnss_logger supports v1.4.0.0 onwards. '
            f'Found version {version}')

    # --- write csv --------------------------------------------------------
    with open(full_path, 'r', errors='replace') as fin, \
         open(csv_path, 'w', newline='') as fout:
        for line in fin:
            if 'Raw,' not in line:
                continue
            line = line.replace('Raw,', '')
            line = line.replace('#', '').replace(' ', '')
            fout.write(line)

    return csv_path


def _read_raw_csv(raw_csv_path):
    """Read CSV file into header list and dict of numpy arrays."""
    with open(raw_csv_path, 'r', errors='replace') as f:
        header_line = f.readline()

    if 'TimeNanos' not in header_line:
        raise ValueError('"TimeNanos" string not found in CSV header')

    header = [h.strip() for h in header_line.strip().split(',')]
    num_fields = len(header)

    # Initialize columns
    C = {h: [] for h in header}

    with open(raw_csv_path, 'r', errors='replace') as f:
        f.readline()  # skip header
        reader = csv.reader(f)
        for row in reader:
            # Pad / trim row to num_fields
            while len(row) < num_fields:
                row.append('')
            row = row[:num_fields]
            for j, h in enumerate(header):
                val = row[j].strip()
                if h in _STRING_FIELDS:
                    C[h].append(np.nan)  # CodeType → NaN
                elif h in _INT64_FIELDS:
                    try:
                        C[h].append(np.int64(val))
                    except (ValueError, OverflowError):
                        C[h].append(np.int64(0))
                else:
                    try:
                        C[h].append(float(val))
                    except ValueError:
                        C[h].append(np.nan)

    # Convert to numpy arrays
    for h in header:
        if h in _INT64_FIELDS:
            C[h] = np.array(C[h], dtype=np.int64)
        else:
            C[h] = np.array(C[h], dtype=float)

    return header, C


def _filter_data(C, data_filter, header):
    """Apply data_filter to C.  Returns (ok, filtered_C)."""
    if not data_filter:
        return True, C

    n = len(C[header[0]])
    mask = np.ones(n, dtype=bool)

    for field, condition in data_filter:
        col = C[field]
        # Use integer array for bitwise-capable fields so that &, |, ~ work
        if np.issubdtype(col.dtype, np.floating):
            int_col = col.astype(np.int64)
        else:
            int_col = col
        # Build a namespace with the field as a variable
        ns = {field: int_col}
        try:
            idx = eval(condition, {"__builtins__": {}}, ns)  # noqa: S307
            idx = np.asarray(idx, dtype=bool)
        except Exception as exc:
            raise ValueError(
                f'Error evaluating data filter condition "{condition}": {exc}') from exc
        mask = mask & idx

    if not np.any(mask):
        print('\nAll measurements removed. Specify dataFilter less strictly.')
        return False, C

    filtered = {h: arr[mask] for h, arr in C.items()}
    return True, filtered


def _pack_gnss_raw(C, header):
    """Pack data dict into gnss_raw, reporting missing fields."""
    gnss_clock_fields = [
        'TimeNanos', 'TimeUncertaintyNanos', 'LeapSecond', 'FullBiasNanos',
        'BiasUncertaintyNanos', 'DriftNanosPerSecond',
        'DriftUncertaintyNanosPerSecond', 'HardwareClockDiscontinuityCount',
        'BiasNanos',
    ]
    gnss_meas_fields = [
        'Cn0DbHz', 'ConstellationType', 'MultipathIndicator',
        'PseudorangeRateMetersPerSecond',
        'PseudorangeRateUncertaintyMetersPerSecond',
        'ReceivedSvTimeNanos', 'ReceivedSvTimeUncertaintyNanos',
        'State', 'Svid',
        'AccumulatedDeltaRangeMeters', 'AccumulatedDeltaRangeUncertaintyMeters',
    ]

    missing = {'ClockFields': [], 'MeasurementFields': []}
    gnss_raw = {}

    for h in header:
        arr = C[h]
        has_finite = (
            np.any(np.isfinite(arr))
            if np.issubdtype(arr.dtype, np.floating)
            else True
        )
        if has_finite:
            gnss_raw[h] = arr
        elif h in gnss_clock_fields:
            missing['ClockFields'].append(h)
        elif h in gnss_meas_fields:
            missing['MeasurementFields'].append(h)

    return gnss_raw, missing


def _check_gnss_clock(gnss_raw, gnss_analysis):
    """Validate clock fields; compute allRxMillis."""
    ok = True
    s_fail = ''
    n = len(gnss_raw.get('ReceivedSvTimeNanos',
                          gnss_raw.get('Svid', np.array([]))))

    if 'TimeNanos' not in gnss_raw:
        s = ' TimeNanos missing from GnssLogger File.'
        print(f'WARNING: {s}')
        s_fail += s
        ok = False

    if 'FullBiasNanos' not in gnss_raw:
        s = 'FullBiasNanos missing from GnssLogger file.'
        print(f'WARNING: {s}, we need it to get the week number')
        s_fail += s
        ok = False

    if 'BiasNanos' not in gnss_raw:
        gnss_raw['BiasNanos'] = np.zeros(n)

    if 'HardwareClockDiscontinuityCount' not in gnss_raw:
        gnss_raw['HardwareClockDiscontinuityCount'] = np.zeros(n, dtype=int)
        print('WARNING: Added HardwareClockDiscontinuityCount=0 because it is '
              'missing from GNSS Logger file')

    if ok:
        fbn = gnss_raw['FullBiasNanos']
        if np.any(fbn < 0) and np.any(fbn > 0):
            raise AssertionError(
                'FullBiasNanos changes sign within log file')
        if np.any(fbn > 0):
            gnss_raw['FullBiasNanos'] = -fbn
            print('WARNING: FullBiasNanos wrong sign. Auto-correcting.')
            gnss_analysis['GnssClockErrors'] += ' FullBiasNanos wrong sign.'

        gnss_raw['allRxMillis'] = np.int64(
            (gnss_raw['TimeNanos'].astype(np.int64)
             - gnss_raw['FullBiasNanos'].astype(np.int64)) // np.int64(1_000_000))

    if not ok:
        gnss_analysis['ApiPassFail'] = 'FAIL ' + s_fail

    return gnss_raw, gnss_analysis


def _report_missing_fields(gnss_analysis, missing):
    """Append missing-field info to gnss_analysis and set pass/fail."""
    if missing['ClockFields']:
        gnss_analysis['GnssClockErrors'] += (
            ' Missing Fields: ' + ', '.join(missing['ClockFields']) + '.')
    if missing['MeasurementFields']:
        gnss_analysis['GnssMeasurementErrors'] += (
            ' Missing Fields: ' + ', '.join(missing['MeasurementFields']) + '.')

    if 'FAIL' not in gnss_analysis.get('ApiPassFail', ''):
        if not missing['ClockFields'] and not missing['MeasurementFields']:
            gnss_analysis['ApiPassFail'] = 'PASS'
        else:
            gnss_analysis['ApiPassFail'] = 'FAIL BECAUSE OF MISSING FIELDS'

    return gnss_analysis
