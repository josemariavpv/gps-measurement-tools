"""Kepler's equation solver.

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

import numpy as np


def kepler(mk, e):
    """Solve Kepler's equation for eccentric anomaly by iteration.

    Parameters
    ----------
    mk : array_like
        Mean anomaly (radians). Scalar or 1-D array.
    e : array_like
        Eccentricity. Scalar or 1-D array of same shape as mk.

    Returns
    -------
    ek : ndarray
        Eccentric anomaly (radians).
    """
    mk = np.atleast_1d(np.asarray(mk, dtype=float)).copy()
    e = np.atleast_1d(np.asarray(e, dtype=float))

    ek = mk.copy()
    iter_count = 0
    max_iter = 20
    err = np.full_like(ek, 1.0)

    while np.any(np.abs(err) > 1e-8) and iter_count < max_iter:
        err = ek - mk - e * np.sin(ek)
        ek = ek - err
        iter_count += 1
        if iter_count == max_iter:
            print("Failed convergence on Kepler's equation.")

    return ek
