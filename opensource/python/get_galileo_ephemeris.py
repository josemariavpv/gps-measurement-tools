"""Download / read daily Galileo + GPS ephemeris from a public RINEX 3 source.

Downloads the daily IGS MGEX mixed navigation file from the BKG FTP mirror
(publicly accessible without authentication).  Caches the file locally for
re-use within the same day.

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

import ftplib
import os

from .day_of_year import day_of_year
from .read_rinex3_nav import read_rinex3_nav


# BKG public FTP mirror for MGEX broadcast navigation files.
# No login credentials are required.
_BKG_FTP_HOST = 'igs.bkg.bund.de'
_BKG_FTP_DIR_TEMPLATE = '/MGEX/BRDC/{year}/{doy:03d}/'
_BKG_FILENAME_TEMPLATE = 'BRDM00DLR_R_{year}{doy:03d}0000_01D_MN.rnx.gz'


def get_galileo_ephemeris(utc_time, dir_name=None):
    """Obtain Galileo (and GPS) ephemeris from the daily IGS MGEX mixed nav file.

    If a matching file already exists in *dir_name* it is read directly;
    otherwise it is downloaded automatically from the BKG public FTP mirror.

    Parameters
    ----------
    utc_time : array_like, shape (6,)
        [year, month, day, hours, minutes, seconds] (UTC).
    dir_name : str, optional
        Local directory for cached / downloaded files.

    Returns
    -------
    gal_eph : list of dict
        Galileo ephemeris records; same field format as GPS ephemeris from
        :func:`read_rinex_nav`.  Empty list on failure.
    gps_eph : list of dict
        GPS ephemeris records parsed from the same mixed nav file.  Empty on
        failure.
    """
    gal_eph = []
    gps_eph = []

    if len(utc_time) != 6:
        raise ValueError('utc_time must have 6 elements')

    year  = int(utc_time[0])
    doy   = int(day_of_year(utc_time))

    filename = _BKG_FILENAME_TEMPLATE.format(year=year, doy=doy)

    if dir_name:
        if not os.path.isdir(dir_name):
            print(f'Warning: directory "{dir_name}" not found')
            return gal_eph, gps_eph
        if not dir_name.endswith(os.sep):
            dir_name = dir_name + os.sep
        full_path = os.path.join(dir_name, filename)
    else:
        full_path = filename

    # ------------------------------------------------------------------
    # Check for an already-decompressed or compressed local copy
    # ------------------------------------------------------------------
    plain_path = full_path[:-3] if full_path.endswith('.gz') else full_path

    if os.path.isfile(plain_path):
        print(f'Reading mixed GNSS nav from "{os.path.basename(plain_path)}" '
              f'in local directory')
        if dir_name:
            print(dir_name)
        return _safe_read(plain_path)

    if os.path.isfile(full_path):
        print(f'Reading mixed GNSS nav from "{filename}" in local directory')
        if dir_name:
            print(dir_name)
        return _safe_read(full_path)

    # ------------------------------------------------------------------
    # Download from BKG FTP
    # ------------------------------------------------------------------
    ftp_dir = _BKG_FTP_DIR_TEMPLATE.format(year=year, doy=doy)
    print(f'\nGetting mixed GNSS nav "{filename}" from {_BKG_FTP_HOST} …')

    try:
        with ftplib.FTP(_BKG_FTP_HOST, timeout=30) as ftp:
            ftp.login()
            ftp.cwd(ftp_dir)
            with open(full_path, 'wb') as fh:
                ftp.retrbinary(f'RETR {filename}', fh.write)
        print(f'Downloaded "{filename}"')
    except Exception as exc:
        print(f'FTP download of mixed nav file failed: {exc}')
        print('Galileo satellites will not be shown in the skyplot.')
        # Clean up any partial file
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass
        return gal_eph, gps_eph

    return _safe_read(full_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_read(file_path):
    """Call read_rinex3_nav, returning ([], []) on any parse error."""
    try:
        gps_eph, gal_eph = read_rinex3_nav(file_path)
        print(f'Mixed nav file: {len(gps_eph)} GPS records, '
              f'{len(gal_eph)} Galileo records.')
        return gal_eph, gps_eph
    except Exception as exc:
        print(f'Warning: could not parse mixed nav file "{file_path}": {exc}')
        return [], []
