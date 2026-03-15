"""Python port of the Google GPS Measurement Tools 'opensource' library.

Provides identical functionality to the original MATLAB code without any
MATLAB dependency.

Quick-start
-----------
    from opensource.python import (
        set_data_filter,
        read_gnss_logger,
        process_gnss_meas,
        gps_wls_pvt,
    )
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

from .sv_label                import sv_label
from .gps_constants           import GpsConstants
from .gnss_thresholds         import GnssThresholds
from .julian_day              import julian_day
from .leap_seconds            import leap_seconds
from .day_of_year             import day_of_year
from .compare_versions        import compare_versions
from .utc2gps                 import utc2gps
from .gps2utc                 import gps2utc
from .kepler                  import kepler
from .read_rinex_nav          import read_rinex_nav
from .get_nasa_hourly_ephemeris import get_nasa_hourly_ephemeris
from .gps_eph2dtsv            import gps_eph2dtsv
from .gps_eph2xyz             import gps_eph2xyz
from .gps_eph2pvt             import gps_eph2pvt
from .flight_time_correction  import flight_time_correction
from .closest_gps_eph         import closest_gps_eph
from .lla2xyz                 import lla2xyz
from .xyz2lla                 import xyz2lla
from .rot_ecef2ned            import rot_ecef2ned
from .lla2ned                 import lla2ned
from .wls_pvt                 import wls_pvt
from .gps_wls_pvt             import gps_wls_pvt
from .set_data_filter         import set_data_filter
from .check_data_filter       import check_data_filter
from .read_gnss_logger        import read_gnss_logger
from .process_gnss_meas       import process_gnss_meas
from .process_adr             import process_adr
from .gps_adr_residuals       import gps_adr_residuals
from .plot_pseudoranges       import plot_pseudoranges
from .plot_pseudorange_rates  import plot_pseudorange_rates
from .plot_cno                import plot_cno
from .plot_pvt                import plot_pvt
from .plot_pvt_states         import plot_pvt_states
from .plot_adr                import plot_adr
from .plot_adr_resids         import plot_adr_resids
from .read_rinex_obs          import read_rinex_obs, get_signal_strength
from .plot_rinex_quality      import (plot_rinex_visibility,
                                      plot_rinex_availability,
                                      plot_rinex_cn0,
                                      plot_rinex_cycle_slips)
try:
    from .gnss_analysis_app       import GnssAnalysisApp
except ImportError:  # tkinter not available (headless / CI environment)
    GnssAnalysisApp = None  # type: ignore[assignment,misc]

__all__ = [
    'GpsConstants', 'GnssThresholds',
    'sv_label',
    'julian_day', 'leap_seconds', 'day_of_year', 'compare_versions',
    'utc2gps', 'gps2utc',
    'kepler',
    'read_rinex_nav', 'get_nasa_hourly_ephemeris',
    'gps_eph2dtsv', 'gps_eph2xyz', 'gps_eph2pvt',
    'flight_time_correction', 'closest_gps_eph',
    'lla2xyz', 'xyz2lla', 'rot_ecef2ned', 'lla2ned',
    'wls_pvt', 'gps_wls_pvt',
    'set_data_filter', 'check_data_filter',
    'read_gnss_logger', 'process_gnss_meas',
    'process_adr', 'gps_adr_residuals',
    'plot_pseudoranges', 'plot_pseudorange_rates', 'plot_cno',
    'plot_pvt', 'plot_pvt_states', 'plot_adr', 'plot_adr_resids',
    'read_rinex_obs', 'get_signal_strength',
    'plot_rinex_visibility', 'plot_rinex_availability',
    'plot_rinex_cn0', 'plot_rinex_cycle_slips',
    'GnssAnalysisApp',
]
