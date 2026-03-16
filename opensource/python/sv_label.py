"""Satellite label utilities.

Converts Android GNSS constellation type + SV ID into a human-readable
label such as ``'G01'``, ``'E06'``, ``'R22'``.
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

# Android GnssStatus.CONSTELLATION_* constants
_CONSTELLATION_CHAR = {
    0: '?',  # UNKNOWN
    1: 'G',  # GPS
    2: 'S',  # SBAS
    3: 'R',  # GLONASS
    4: 'J',  # QZSS
    5: 'C',  # BEIDOU
    6: 'E',  # GALILEO
    7: 'I',  # IRNSS / NavIC
}


def sv_label(constellation_type, svid):
    """Return a satellite label like ``'G01'``, ``'E05'``, ``'R22'``.

    Parameters
    ----------
    constellation_type : int
        Android ``GnssStatus.CONSTELLATION_*`` constant.
    svid : int
        Satellite vehicle ID.

    Returns
    -------
    str
        Label of the form ``'<char><NN>'`` (e.g. ``'G01'``, ``'E06'``).
    """
    char = _CONSTELLATION_CHAR.get(int(constellation_type), '?')
    return f'{char}{int(svid):02d}'
