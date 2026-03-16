"""Validate a data filter for use with ReadGnssLogger (CheckDataFilter).

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


def check_data_filter(data_filter, header=None):
    """Check that data_filter is defined correctly.

    Parameters
    ----------
    data_filter : list of (str, str) or None
        Each entry is (field_name, condition_string), where condition_string
        is a valid Python expression containing field_name as a variable.
    header : list of str, optional
        Column names from the GnssLogger CSV header.  When provided each
        field_name in data_filter is checked against the header.

    Returns
    -------
    ok : bool
        True if the filter is valid.
    """
    if data_filter is None or len(data_filter) == 0:
        return True

    if not isinstance(data_filter, list):
        raise ValueError('data_filter must be a list of (field, condition) pairs')

    for i, entry in enumerate(data_filter):
        if len(entry) != 2:
            raise ValueError(
                f'data_filter[{i}] must be a 2-element tuple (field, condition)')
        field, condition = entry
        if not isinstance(field, str):
            raise ValueError(f'data_filter[{i}][0] is not a string')
        if not isinstance(condition, str):
            raise ValueError(f'data_filter[{i}][1] is not a string')
        if field not in condition:
            raise ValueError(
                f'data_filter[{i}][0] string "{field}" '
                f'not found in data_filter[{i}][1]')

    if header is not None:
        for field, _ in data_filter:
            if field not in header:
                raise ValueError(
                    f'data_filter value "{field}" has no match in log file header')

    return True
