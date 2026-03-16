"""Download / read NASA hourly RINEX navigation ephemeris (GetNasaHourlyEphemeris).

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
import ftplib
import subprocess

from .gps_constants import GpsConstants
from .gnss_thresholds import GnssThresholds
from .day_of_year import day_of_year
from .utc2gps import utc2gps
from .read_rinex_nav import read_rinex_nav


def get_nasa_hourly_ephemeris(utc_time, dir_name=None):
    """Obtain an hourly GPS ephemeris file from NASA CDDIS.

    If a file with valid ephemeris for at least ``GnssThresholds.MINNUMGPSEPH``
    satellites already exists in *dir_name*, it is read directly; otherwise the
    file is downloaded automatically via FTP.

    Parameters
    ----------
    utc_time : array_like, shape (6,)
        [year, month, day, hours, minutes, seconds] (UTC).
    dir_name : str, optional
        Local directory for cached / downloaded ephemeris files.

    Returns
    -------
    all_gps_eph : list of dict
        GPS ephemeris records; see read_rinex_nav().
    all_glo_eph : list
        GLONASS ephemeris (currently not populated, returns ``[]``).
    """
    all_gps_eph = []
    all_glo_eph = []

    ok, dir_name = _check_inputs(utc_time, dir_name)
    if not ok:
        return all_gps_eph, all_glo_eph

    year4 = int(utc_time[0])
    year2 = year4 % 100
    day   = int(day_of_year(utc_time))

    hourly_z_file = f'hour{day:03d}0.{year2:02d}n.Z'
    eph_filename  = hourly_z_file[:-2]
    full_eph      = os.path.join(dir_name, eph_filename) if dir_name else eph_filename

    b_got_gps_eph = False
    if os.path.isfile(full_eph):
        print(f'Reading GPS ephemeris from "{eph_filename}" in local directory')
        if dir_name:
            print(dir_name)
        all_gps_eph, _ = read_rinex_nav(full_eph)
        _, fct_secs = utc2gps(utc_time)
        fct_seconds = float(fct_secs[0])

        eph_age = [
            e['GPS_Week'] * GpsConstants.WEEKSEC + e['Toe'] - fct_seconds
            for e in all_gps_eph
        ]
        i_fresh = [
            abs(age) < GpsConstants.EPHVALIDSECONDS
            and not e.get('health', 0)
            for age, e in zip(eph_age, all_gps_eph)
        ]
        good_svs = set(
            all_gps_eph[i]['PRN']
            for i, ok_flag in enumerate(i_fresh) if ok_flag
        )
        if len(good_svs) >= GnssThresholds.MINNUMGPSEPH:
            b_got_gps_eph = True

    if not b_got_gps_eph:
        if dir_name and '~' in dir_name:
            print(f'\nYou set: dir_name = "{dir_name}"')
            print('To download ephemeris from FTP, specify dir_name '
                  'with full path and no tilde "~"')
            print(f'Change dir_name or download ephemeris file '
                  f'"{hourly_z_file}" by hand.')
            return all_gps_eph, all_glo_eph

        ftp_site  = 'cddis.gsfc.nasa.gov'
        hourly_dir = f'/gnss/data/hourly/{year4:4d}/{day:03d}/'
        print(f'\nGetting GPS ephemeris "{hourly_z_file}" from NASA FTP ...')

        try:
            with ftplib.FTP(ftp_site) as ftp:
                ftp.login()
                ftp.cwd(hourly_dir)
                z_path = os.path.join(dir_name or '.', hourly_z_file)
                with open(z_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {hourly_z_file}', f.write)
        except Exception as exc:
            print('\nAutomatic FTP download failed.')
            print(f'Please go to ftp://{ftp_site}/')
            print(f'and get this file: {hourly_dir}{hourly_z_file}')
            if dir_name:
                print(f'Extract contents to: {dir_name}')
            print('To unzip the *.Z file, see http://www.gzip.org/')
            print('Then run this function again; it will read from that directory')
            return all_gps_eph, all_glo_eph

        # Decompress the .Z file
        try:
            subprocess.run(['uncompress', z_path], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                subprocess.run(['gunzip', z_path], check=True)
            except Exception as exc2:
                print(f'\nError: could not decompress "{hourly_z_file}": {exc2}')
                print('Unzip by hand. See http://www.gzip.org/')
                print('Then run this function again.')
                return all_gps_eph, all_glo_eph

        print(f'\nSuccessfully downloaded ephemeris file "{eph_filename}"')
        if dir_name:
            print(f'to: {dir_name}')

    all_gps_eph, _ = read_rinex_nav(full_eph)
    return all_gps_eph, all_glo_eph


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_inputs(utc_time, dir_name):
    """Validate inputs and normalise dir_name."""
    if len(utc_time) != 6:
        raise ValueError('utc_time must have 6 elements')

    if not dir_name:
        return True, dir_name

    if not os.path.isdir(dir_name):
        print(f'Error: directory "{dir_name}" not found')
        return False, dir_name

    if not dir_name.endswith(os.sep):
        dir_name = dir_name + os.sep

    return True, dir_name
