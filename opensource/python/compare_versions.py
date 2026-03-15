"""Version comparison utility.

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


def compare_versions(v, w):
    """Compare version v to version w.

    Parameters
    ----------
    v, w : sequence of int
        Version tuples of equal length, e.g. (1, 4, 0, 0).

    Returns
    -------
    s : str
        'before' if v < w, 'equal' if v == w, 'after' if v > w.
    """
    if len(v) != len(w):
        raise ValueError('The two inputs must be sequences of the same length')

    for vi, wi in zip(v, w):
        if vi < wi:
            return 'before'
        elif vi > wi:
            return 'after'
    return 'equal'
