"""Set the data filter for ReadGnssLogger (SetDataFilter).

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


def set_data_filter():
    """Return the default data filter for use with read_gnss_logger().

    The data filter is a list of (field_name, condition_string) pairs.
    Each condition_string is a Python boolean expression in which the
    field_name appears as a variable name referring to a numpy array.

    Returns
    -------
    data_filter : list of (str, str)
        Filter conditions applied by check_data_filter / read_gnss_logger.

    Examples
    --------
    To add an additional filter (remove SV 23):
        data_filter.append(('Svid', 'Svid != 23'))

    To keep only Svid 2, 5, 10, 17:
        data_filter.append(('Svid',
            '(Svid==2) | (Svid==5) | (Svid==10) | (Svid==17)'))
    """
    data_filter = []

    # Filter out FullBiasNanos == 0
    data_filter.append(('FullBiasNanos', 'FullBiasNanos != 0'))

    # Filter for fine time measurements only  <=> uncertainty < 10 ms = 1e7 ns
    # For Nexus 5x and 6p this field is not filled, so this is commented out:
    # data_filter.append(('BiasUncertaintyNanos', 'BiasUncertaintyNanos < 1e7'))

    # Limit to GPS only (ConstellationType == 1):
    # ConstellationType constants (Android HAL gps.h):
    #   UNKNOWN=0, GPS=1, SBAS=2, GLONASS=3, QZSS=4, BEIDOU=5, GALILEO=6
    data_filter.append(('ConstellationType', 'ConstellationType == 1'))

    # Filter on GnssMeasurementState: CODE_LOCK (bit 0) & TOW_DECODED (bit 3)
    # Note: cast each bitmask to bool before combining, so that e.g.
    # (15 & 1) != 0  AND  (15 & 8) != 0  evaluates correctly.
    data_filter.append(
        ('State',
         '((State & (1 << 0)) != 0) & ((State & (1 << 3)) != 0)'))

    return data_filter
