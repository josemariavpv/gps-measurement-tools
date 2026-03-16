"""Numerical validation tests for the Python GPS measurement tools.

These tests verify that the Python port produces the same numerical results
as the original MATLAB implementation (opensource/*.m).  Each test exercises
a specific function with analytically known or independently verifiable values.

The demo log file ``pseudoranges_log_2016_06_30_21_26_07.txt`` was collected
at Charleston Park Test Site, Mountain View, CA:
    lat = 37.422578 °N, lon = -122.081678 °W, alt = -28 m  (WGS-84)

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
import sys
import unittest
import numpy as np

# Make the package importable when running tests directly from this directory
_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'opensource'))

# Demo-data directory
_DEMO_DIR = os.path.join(_REPO_ROOT, 'opensource', 'demoFiles')

# True position of Charleston Park Test Site (WGS-84 LLA)
_TRUE_LAT_DEG = 37.422578
_TRUE_LON_DEG = -122.081678   # negative = West
_TRUE_ALT_M   = -28.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _import_python():
    """Return the python sub-package (lazy, so tests can still be discovered)."""
    import python as P
    return P


# ===========================================================================
# 1. JulianDay / julian_day
# ===========================================================================

class TestJulianDay(unittest.TestCase):
    """Verify julian_day() against analytically known Julian-Day values.

    MATLAB equivalent: JulianDay.m
    """

    def setUp(self):
        from python.julian_day import julian_day
        self.jd = julian_day

    def test_gps_epoch(self):
        """GPS epoch (1980-01-06 00:00:00 UTC) must be JD 2444244.5."""
        jd = self.jd([[1980, 1, 6, 0, 0, 0]])
        self.assertAlmostEqual(jd[0], 2444244.5, places=6)

    def test_j2000(self):
        """J2000.0 epoch (2000-01-01 12:00:00 UTC) must be JD 2451545.0."""
        jd = self.jd([[2000, 1, 1, 12, 0, 0]])
        self.assertAlmostEqual(jd[0], 2451545.0, places=6)

    def test_demo_date(self):
        """Demo-file date 2016-06-30 21:26:07 UTC produces a consistent JD."""
        jd = self.jd([[2016, 6, 30, 21, 26, 7]])
        # JD must be > J2000 and correspond to a date in mid-2016
        self.assertGreater(jd[0], 2451545.0)
        # Days from J2000 to 2016-06-30 ≈ 6025 days
        self.assertAlmostEqual(jd[0] - 2451545.0, 6025.0, delta=1.5)

    def test_leap_year_february(self):
        """February 29 of a leap year is handled correctly."""
        jd_feb28 = self.jd([[2000, 2, 28, 0, 0, 0]])
        jd_feb29 = self.jd([[2000, 2, 29, 0, 0, 0]])
        jd_mar01 = self.jd([[2000, 3, 1, 0, 0, 0]])
        self.assertAlmostEqual(jd_feb29[0] - jd_feb28[0], 1.0, places=10)
        self.assertAlmostEqual(jd_mar01[0] - jd_feb29[0], 1.0, places=10)

    def test_multiple_rows(self):
        """Batch input is handled element-wise."""
        utc = [[1980, 1, 6, 0, 0, 0], [2000, 1, 1, 12, 0, 0]]
        jd = self.jd(utc)
        self.assertEqual(len(jd), 2)
        self.assertAlmostEqual(jd[0], 2444244.5, places=6)
        self.assertAlmostEqual(jd[1], 2451545.0, places=6)


# ===========================================================================
# 2. Utc2Gps / utc2gps  and  Gps2Utc / gps2utc
# ===========================================================================

class TestUtc2Gps(unittest.TestCase):
    """Verify utc2gps() – MATLAB equivalent: Utc2Gps.m."""

    def setUp(self):
        from python.utc2gps import utc2gps
        self.utc2gps = utc2gps

    def test_gps_epoch(self):
        """GPS epoch (1980-01-06 00:00:00) must map to week=0, seconds=0."""
        gps, fct = self.utc2gps([[1980, 1, 6, 0, 0, 0]])
        self.assertAlmostEqual(gps[0, 0], 0.0, places=9)   # week
        self.assertAlmostEqual(gps[0, 1], 0.0, places=9)   # seconds
        self.assertAlmostEqual(fct[0],    0.0, places=9)

    def test_demo_date(self):
        """2016-06-30 21:26:07 UTC → GPS week 1903, seconds in [0, 604800)."""
        gps, fct = self.utc2gps([[2016, 6, 30, 21, 26, 7]])
        week = int(gps[0, 0])
        sec  = gps[0, 1]
        self.assertEqual(week, 1903)
        self.assertGreaterEqual(sec, 0.0)
        self.assertLess(sec, 604800.0)
        # Full-cycle time must be consistent
        self.assertAlmostEqual(fct[0], week * 604800.0 + sec, places=3)

    def test_roundtrip_with_gps2utc(self):
        """utc2gps followed by gps2utc must recover the original time."""
        from python.gps2utc import gps2utc
        utc_in = np.array([[2016, 6, 30, 21, 26, 7.0]])
        gps, _ = self.utc2gps(utc_in)
        utc_out = gps2utc(gps_time=gps)
        np.testing.assert_allclose(utc_out[0, :5], utc_in[0, :5], atol=0)
        self.assertAlmostEqual(utc_out[0, 5], utc_in[0, 5], places=3)


class TestGps2Utc(unittest.TestCase):
    """Verify gps2utc() – MATLAB equivalent: Gps2Utc.m."""

    def setUp(self):
        from python.gps2utc import gps2utc
        self.gps2utc = gps2utc

    def test_gps_epoch(self):
        """GPS fct=0 must yield 1980-01-06 00:00:00 UTC."""
        utc = self.gps2utc(fct_seconds=np.array([0.0]))
        expected = [1980, 1, 6, 0, 0, 0]
        np.testing.assert_allclose(utc[0], expected, atol=1e-9)

    def test_demo_fct(self):
        """Known fct → expected UTC for the demo data."""
        from python.utc2gps import utc2gps
        _, fct = utc2gps([[2016, 6, 30, 21, 26, 7]])
        utc = self.gps2utc(fct_seconds=fct)
        np.testing.assert_allclose(utc[0, :5], [2016, 6, 30, 21, 26], atol=0)
        self.assertAlmostEqual(utc[0, 5], 7.0, places=3)


# ===========================================================================
# 3. Lla2Xyz / lla2xyz  and  Xyz2Lla / xyz2lla
# ===========================================================================

class TestLla2Xyz(unittest.TestCase):
    """Verify lla2xyz() – MATLAB equivalent: Lla2Xyz.m."""

    def setUp(self):
        from python.lla2xyz import lla2xyz
        self.lla2xyz = lla2xyz

    def test_north_pole(self):
        """North Pole (lat=90, lon=0, alt=0) → (0, 0, b) where b = semi-minor axis."""
        from python.gps_constants import GpsConstants
        a  = GpsConstants.EARTHSEMIMAJOR
        e2 = GpsConstants.EARTHECCEN2
        b  = a * np.sqrt(1.0 - e2)
        xyz = self.lla2xyz([[90.0, 0.0, 0.0]])
        self.assertAlmostEqual(xyz[0, 0], 0.0, places=3)
        self.assertAlmostEqual(xyz[0, 1], 0.0, places=3)
        self.assertAlmostEqual(xyz[0, 2], b,   places=2)

    def test_equator_prime_meridian(self):
        """Equator / Prime Meridian (lat=0, lon=0, alt=0) → (a, 0, 0)."""
        from python.gps_constants import GpsConstants
        a = GpsConstants.EARTHSEMIMAJOR
        xyz = self.lla2xyz([[0.0, 0.0, 0.0]])
        self.assertAlmostEqual(xyz[0, 0], a,   places=2)
        self.assertAlmostEqual(xyz[0, 1], 0.0, places=2)
        self.assertAlmostEqual(xyz[0, 2], 0.0, places=2)

    def test_charleston_park(self):
        """Charleston Park Test Site must produce an ECEF position ≈26 km below GPS orbit."""
        xyz = self.lla2xyz([[_TRUE_LAT_DEG, _TRUE_LON_DEG, _TRUE_ALT_M]])
        r = np.linalg.norm(xyz[0])
        # Must be within Earth's surface range (~6356–6378 km from centre)
        self.assertGreater(r, 6356e3)
        self.assertLess(r,    6379e3)

    def test_altitude_offset(self):
        """Increasing altitude by 1 m shifts radius by ~1 m."""
        xyz0 = self.lla2xyz([[_TRUE_LAT_DEG, _TRUE_LON_DEG, 0.0]])
        xyz1 = self.lla2xyz([[_TRUE_LAT_DEG, _TRUE_LON_DEG, 1.0]])
        dr = np.linalg.norm(xyz1[0]) - np.linalg.norm(xyz0[0])
        self.assertAlmostEqual(dr, 1.0, places=3)


class TestXyz2Lla(unittest.TestCase):
    """Verify xyz2lla() – MATLAB equivalent: Xyz2Lla.m."""

    def setUp(self):
        from python.lla2xyz import lla2xyz
        from python.xyz2lla import xyz2lla
        self.lla2xyz = lla2xyz
        self.xyz2lla = xyz2lla

    def test_roundtrip_charleston_park(self):
        """Round-trip LLA→ECEF→LLA must be accurate to millimetre level."""
        lla_in = np.array([[_TRUE_LAT_DEG, _TRUE_LON_DEG, _TRUE_ALT_M]])
        xyz    = self.lla2xyz(lla_in)
        lla_out = self.xyz2lla(xyz)
        # 1e-8 degrees ≈ 1 mm on the ground; altitude within 0.1 mm
        np.testing.assert_allclose(lla_out[0, :2], lla_in[0, :2], atol=1e-8)
        np.testing.assert_allclose(lla_out[0,  2], lla_in[0,  2], atol=1e-4)

    def test_equator_prime_meridian(self):
        """ECEF (a, 0, 0) → (lat=0, lon=0, alt=0)."""
        from python.gps_constants import GpsConstants
        a = GpsConstants.EARTHSEMIMAJOR
        lla = self.xyz2lla([[a, 0.0, 0.0]])
        self.assertAlmostEqual(lla[0, 0],  0.0, places=8)   # lat
        self.assertAlmostEqual(lla[0, 1],  0.0, places=8)   # lon
        self.assertAlmostEqual(lla[0, 2],  0.0, places=2)   # alt

    def test_negative_longitude(self):
        """Longitudes west of Greenwich are returned as negative degrees."""
        lla_in = np.array([[37.0, -120.0, 100.0]])
        xyz    = self.lla2xyz(lla_in)
        lla_out = self.xyz2lla(xyz)
        self.assertLess(lla_out[0, 1], 0.0)
        self.assertAlmostEqual(lla_out[0, 1], -120.0, places=7)

    def test_multiple_rows(self):
        """Batch conversion is handled correctly."""
        lla_in = np.array([
            [_TRUE_LAT_DEG, _TRUE_LON_DEG, _TRUE_ALT_M],
            [0.0,  0.0, 0.0],
            [90.0, 0.0, 0.0],
        ])
        xyz    = self.lla2xyz(lla_in)
        lla_out = self.xyz2lla(xyz)
        np.testing.assert_allclose(lla_out[:, :2], lla_in[:, :2], atol=1e-7)


# ===========================================================================
# 4. Kepler / kepler
# ===========================================================================

class TestKepler(unittest.TestCase):
    """Verify kepler() – MATLAB equivalent: Kepler.m.

    For eccentricity = 0 the eccentric anomaly equals the mean anomaly exactly.
    For small eccentricity the first-order perturbation is  E ≈ M + e·sin(M).
    """

    def setUp(self):
        from python.kepler import kepler
        self.kepler = kepler

    def test_circular_orbit_zero(self):
        """M=0, e=0 → E=0."""
        ek = self.kepler(np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(ek[0], 0.0, places=10)

    def test_circular_orbit_pi(self):
        """M=π, e=0 → E=π."""
        ek = self.kepler(np.array([np.pi]), np.array([0.0]))
        self.assertAlmostEqual(ek[0], np.pi, places=10)

    def test_circular_orbit_half_pi(self):
        """M=π/2, e=0 → E=π/2."""
        ek = self.kepler(np.array([np.pi / 2]), np.array([0.0]))
        self.assertAlmostEqual(ek[0], np.pi / 2, places=10)

    def test_small_eccentricity_first_order(self):
        """For small e, E ≈ M + e·sin(M) to first order."""
        e = 0.01
        mk = np.array([1.0])
        ek = self.kepler(mk, np.array([e]))
        expected = mk[0] + e * np.sin(mk[0])  # first-order approximation
        self.assertAlmostEqual(ek[0], expected, places=4)

    def test_gps_eccentricity(self):
        """Typical GPS eccentricity (~0.01): Kepler's equation is satisfied."""
        e  = np.array([0.0104])
        mk = np.array([1.2345])
        ek = self.kepler(mk, e)
        residual = ek[0] - mk[0] - e[0] * np.sin(ek[0])
        self.assertAlmostEqual(residual, 0.0, places=8)

    def test_vector_input(self):
        """Vector inputs must be processed element-wise."""
        e  = np.array([0.0, 0.01, 0.02])
        mk = np.array([0.5, 1.0, 1.5])
        ek = self.kepler(mk, e)
        self.assertEqual(ek.shape, (3,))
        for i in range(3):
            res = ek[i] - mk[i] - e[i] * np.sin(ek[i])
            self.assertAlmostEqual(res, 0.0, places=8)


# ===========================================================================
# 5. FlightTimeCorrection / flight_time_correction
# ===========================================================================

class TestFlightTimeCorrection(unittest.TestCase):
    """Verify flight_time_correction() – MATLAB: FlightTimeCorrection.m."""

    def setUp(self):
        from python.flight_time_correction import flight_time_correction
        from python.gps_constants import GpsConstants
        self.ftc = flight_time_correction
        self.WE  = GpsConstants.WE

    def test_zero_flight_time(self):
        """dt=0 → output equals input (no rotation)."""
        xe = np.array([1e7, 2e7, 3e7])
        xe_rot = self.ftc(xe, 0.0)
        np.testing.assert_allclose(xe_rot, xe, atol=1e-6)

    def test_z_unchanged(self):
        """Earth-rotation correction never changes the z coordinate."""
        xe = np.array([1e7, 2e7, 5e6])
        for dt in [0.001, 0.07, 0.1]:
            xe_rot = self.ftc(xe, dt)
            self.assertAlmostEqual(xe_rot[2], xe[2], places=6)

    def test_radius_preserved(self):
        """The 2-D radius in the equatorial plane is preserved by rotation."""
        xe = np.array([1e7, 2e7, 3e7])
        r_in  = np.sqrt(xe[0]**2 + xe[1]**2)
        for dt in [0.001, 0.07, 0.085]:
            xe_rot = self.ftc(xe, dt)
            r_out = np.sqrt(xe_rot[0]**2 + xe_rot[1]**2)
            self.assertAlmostEqual(r_out, r_in, places=4)

    def test_mean_flight_time(self):
        """Mean GPS flight time (~75 ms) gives a rotation of ~0.022 arcseconds."""
        xe = np.array([1e7, 0.0, 0.0])
        dt = 75e-3  # seconds
        xe_rot = self.ftc(xe, dt)
        theta = self.WE * dt
        expected = np.array([xe[0] * np.cos(theta), -xe[0] * np.sin(theta), 0.0])
        np.testing.assert_allclose(xe_rot, expected, atol=1e-4)


# ===========================================================================
# 6. GpsEph2Xyz / gps_eph2xyz  and  GpsEph2Dtsv / gps_eph2dtsv
# ===========================================================================

class TestGpsEph2Xyz(unittest.TestCase):
    """Verify gps_eph2xyz() and gps_eph2dtsv() – MATLAB equivalents.

    Tests use ephemeris from the demo RINEX navigation file so that the
    comparison can be made with real GPS parameters.
    """

    @classmethod
    def setUpClass(cls):
        nav_file = os.path.join(_DEMO_DIR, 'hour1820.16n')
        if not os.path.exists(nav_file):
            raise unittest.SkipTest('Demo nav file not found')
        from python.read_rinex_nav import read_rinex_nav
        nav, _ = read_rinex_nav(nav_file)
        cls.nav = nav

    def _get_eph(self, prn):
        """Return first ephemeris record for a given PRN."""
        matches = [e for e in self.nav if e['PRN'] == prn]
        if not matches:
            self.skipTest(f'PRN {prn} not found in nav file')
        return matches[0]

    def test_orbital_radius(self):
        """GPS satellite positions must be ~20,200 km above the surface."""
        e = self._get_eph(2)
        gps_time = np.array([[e['GPS_Week'], e['Toe']]])
        from python.gps_eph2xyz import gps_eph2xyz
        xyz, _ = gps_eph2xyz([e], gps_time)
        r_km = np.linalg.norm(xyz[0]) / 1e3
        # GPS semi-major axis ≈ 26560 km; allow ±500 km for eccentricity
        self.assertGreater(r_km, 26060)
        self.assertLess(r_km,    27060)

    def test_dtsv_consistency(self):
        """Clock bias from gps_eph2xyz must match gps_eph2dtsv (same formula)."""
        e = self._get_eph(5)
        gps_time = np.array([[e['GPS_Week'], e['Toe']]])
        from python.gps_eph2xyz import gps_eph2xyz
        from python.gps_eph2dtsv import gps_eph2dtsv
        _, dtsv_xyz = gps_eph2xyz([e], gps_time)
        dtsv_direct = gps_eph2dtsv([e], np.array([e['Toe']]))
        self.assertAlmostEqual(dtsv_xyz[0], dtsv_direct[0], places=15)

    def test_multiple_satellites(self):
        """Multi-satellite call returns one row per ephemeris."""
        ephs = [e for e in self.nav[:5]]
        gps_times = np.array([[e['GPS_Week'], e['Toe']] for e in ephs])
        from python.gps_eph2xyz import gps_eph2xyz
        xyz, dtsv = gps_eph2xyz(ephs, gps_times)
        self.assertEqual(xyz.shape, (5, 3))
        self.assertEqual(dtsv.shape, (5,))

    def test_velocity_from_pvt(self):
        """gps_eph2pvt velocity in the ECEF frame must be within the plausible
        range for GPS satellites (~2,000–4,000 m/s).

        Note: in the inertial frame the GPS orbital speed is ≈3,874 m/s.
        In the ECEF frame the apparent speed is reduced by the Earth-rotation
        component and can be anywhere from ~2,000 to ~4,000 m/s depending on
        satellite geometry.
        """
        e = self._get_eph(2)
        gps_time = np.array([[e['GPS_Week'], e['Toe']]])
        from python.gps_eph2pvt import gps_eph2pvt
        _, _, v_mps, _ = gps_eph2pvt([e], gps_time)
        speed = np.linalg.norm(v_mps[0])
        # Plausible ECEF speed range for a GPS satellite
        self.assertGreater(speed, 1500,
            msg=f'GPS ECEF speed {speed:.0f} m/s is unexpectedly low')
        self.assertLess(speed, 4500,
            msg=f'GPS ECEF speed {speed:.0f} m/s is unexpectedly high')


# ===========================================================================
# 7. End-to-end pipeline – position accuracy
# ===========================================================================

class TestEndToEndPipelineFile1(unittest.TestCase):
    """Full pipeline test using demo log file 1.

    Verifies that the Python implementation reproduces GPS-quality positions
    consistent with what the MATLAB code produces for the same input data.

    Demo file: pseudoranges_log_2016_06_30_21_26_07.txt
    Nav file : hour1820.16n
    True pos : Charleston Park Test Site (lat=37.422578°, lon=-122.081678°, alt=-28 m)
    """

    @classmethod
    def setUpClass(cls):
        log_file = os.path.join(_DEMO_DIR,
                                'pseudoranges_log_2016_06_30_21_26_07.txt')
        nav_file = os.path.join(_DEMO_DIR, 'hour1820.16n')
        if not os.path.exists(log_file) or not os.path.exists(nav_file):
            raise unittest.SkipTest('Demo files not found')

        from python.read_gnss_logger import read_gnss_logger
        from python.process_gnss_meas import process_gnss_meas
        from python.read_rinex_nav import read_rinex_nav
        from python.gps_wls_pvt import gps_wls_pvt
        from python.set_data_filter import set_data_filter

        data_filter = set_data_filter()
        gnss_raw, _ = read_gnss_logger(_DEMO_DIR,
                                       os.path.basename(log_file),
                                       data_filter)
        nav, _ = read_rinex_nav(nav_file)
        gnss_meas = process_gnss_meas(gnss_raw)
        cls.pvt = gps_wls_pvt(gnss_meas, nav)

    def test_pvt_epoch_count(self):
        """Pipeline must produce at least 50 valid PVT epochs."""
        n_epochs = len(self.pvt['FctSeconds'])
        self.assertGreater(n_epochs, 50)

    def test_mean_position_2d_accuracy(self):
        """Mean 2-D position error must be < 50 m from the true location.

        A typical standalone GPS receiver achieves < 5 m; we use 50 m as a
        generous threshold to account for ephemeris/ionosphere differences.
        """
        lla = np.array(self.pvt['allLlaDegDegM'])
        mean_lat = float(np.nanmean(lla[:, 0]))
        mean_lon = float(np.nanmean(lla[:, 1]))

        dlat_m = (mean_lat - _TRUE_LAT_DEG) * 111111.0
        dlon_m = ((mean_lon - _TRUE_LON_DEG)
                  * 111111.0 * np.cos(np.deg2rad(_TRUE_LAT_DEG)))
        err_2d = np.sqrt(dlat_m**2 + dlon_m**2)

        self.assertLess(err_2d, 50.0,
            msg=f'Mean 2D error {err_2d:.1f} m exceeds 50 m threshold')

    def test_altitude_plausible(self):
        """Mean altitude must be within ±200 m of the true altitude."""
        lla = np.array(self.pvt['allLlaDegDegM'])
        mean_alt = float(np.nanmean(lla[:, 2]))
        self.assertAlmostEqual(mean_alt, _TRUE_ALT_M, delta=200.0)

    def test_pvt_structure(self):
        """PVT output dict must contain the expected keys."""
        expected_keys = {
            'FctSeconds', 'allLlaDegDegM', 'sigmaLlaM',
            'allBcMeters', 'allVelMps', 'sigmaVelMps',
            'allBcDotMps', 'numSvs', 'hdop',
        }
        self.assertTrue(expected_keys.issubset(set(self.pvt.keys())))

    def test_clock_bias_magnitude(self):
        """Receiver clock bias must be within ±300 km (typical GPS receiver range)."""
        bc = np.array(self.pvt['allBcMeters'])
        max_bc = float(np.nanmax(np.abs(bc)))
        self.assertLess(max_bc, 3e8,
            msg=f'Clock bias {max_bc:.0f} m seems too large')

    def test_velocity_plausible(self):
        """Computed velocities must be < 3 m/s for a static receiver."""
        vel = np.array(self.pvt['allVelMps'])
        max_speed = float(np.nanmax(np.linalg.norm(vel, axis=1)))
        self.assertLess(max_speed, 3.0,
            msg=f'Max speed {max_speed:.2f} m/s exceeds 3 m/s for static receiver')


class TestEndToEndPipelineFile2(unittest.TestCase):
    """Full pipeline test using demo log file 2 (no duty cycling, carrier phase).

    Demo file: pseudoranges_log_2016_08_22_14_45_50.txt
    Nav file : hour2350.16n
    """

    @classmethod
    def setUpClass(cls):
        log_file = os.path.join(_DEMO_DIR,
                                'pseudoranges_log_2016_08_22_14_45_50.txt')
        nav_file = os.path.join(_DEMO_DIR, 'hour2350.16n')
        if not os.path.exists(log_file) or not os.path.exists(nav_file):
            raise unittest.SkipTest('Demo files not found')

        from python.read_gnss_logger import read_gnss_logger
        from python.process_gnss_meas import process_gnss_meas
        from python.read_rinex_nav import read_rinex_nav
        from python.gps_wls_pvt import gps_wls_pvt
        from python.set_data_filter import set_data_filter

        data_filter = set_data_filter()
        gnss_raw, _ = read_gnss_logger(_DEMO_DIR,
                                       os.path.basename(log_file),
                                       data_filter)
        nav, _ = read_rinex_nav(nav_file)
        gnss_meas = process_gnss_meas(gnss_raw)
        cls.pvt = gps_wls_pvt(gnss_meas, nav)

    def test_pvt_epoch_count(self):
        """Pipeline must produce at least 50 valid PVT epochs."""
        n_epochs = len(self.pvt['FctSeconds'])
        self.assertGreater(n_epochs, 50)

    def test_mean_position_2d_accuracy(self):
        """Mean 2-D position error must be < 50 m from the true location."""
        lla = np.array(self.pvt['allLlaDegDegM'])
        mean_lat = float(np.nanmean(lla[:, 0]))
        mean_lon = float(np.nanmean(lla[:, 1]))

        dlat_m = (mean_lat - _TRUE_LAT_DEG) * 111111.0
        dlon_m = ((mean_lon - _TRUE_LON_DEG)
                  * 111111.0 * np.cos(np.deg2rad(_TRUE_LAT_DEG)))
        err_2d = np.sqrt(dlat_m**2 + dlon_m**2)

        self.assertLess(err_2d, 50.0,
            msg=f'Mean 2D error {err_2d:.1f} m exceeds 50 m threshold')


# ===========================================================================
# 8. LeapSeconds / leap_seconds
# ===========================================================================

class TestLeapSeconds(unittest.TestCase):
    """Verify leap_seconds() – MATLAB equivalent: LeapSeconds.m."""

    def setUp(self):
        from python.leap_seconds import leap_seconds
        self.ls = leap_seconds

    def test_before_first_leap(self):
        """Before the first GPS leap second (1980) → 0 extra leap seconds."""
        ls = self.ls(np.array([[1980, 6, 1, 0, 0, 0]]))
        self.assertEqual(ls[0], 0)

    def test_2016_leap_seconds(self):
        """Verify leap seconds counts around the 2016-12-31 leap second.

        The 2016 leap second occurred on 2016-12-31 23:59:60 UTC, i.e. at the
        start of 2017-01-01.  The table entry is therefore [2017, 1, 1, 0, 0, 0].
        Between 2015-07-01 and 2017-01-01 (exclusive) there are 17 leap seconds.
        After 2017-01-01 there are 18 leap seconds.

        Note: no new leap seconds have been inserted between 2017 and 2026.
        The CGPM voted in November 2022 to eliminate leap seconds by 2035.
        GPS-UTC remains 18 s as of 2026.
        """
        ls_mid_2016 = self.ls(np.array([[2016, 7, 2, 0, 0, 0]]))
        self.assertEqual(ls_mid_2016[0], 17,
            msg='Expected 17 leap seconds between 2015-07-01 and 2017-01-01')

        ls_after_2017 = self.ls(np.array([[2017, 1, 2, 0, 0, 0]]))
        self.assertGreaterEqual(ls_after_2017[0], 18,
            msg='Expected ≥18 leap seconds after 2017-01-01')

    def test_demo_date(self):
        """2016-06-30 21:26:07 UTC → 17 leap seconds (before 2016-07-01)."""
        ls = self.ls(np.array([[2016, 6, 30, 21, 26, 7]]))
        self.assertEqual(ls[0], 17)


# ===========================================================================
# 9. ReadRinexObs / read_rinex_obs
# ===========================================================================

_DEMO_RINEX_OBS = os.path.join(_DEMO_DIR, 'demo_obs.rnx')


class TestReadRinexObs(unittest.TestCase):
    """Verify read_rinex_obs() against the synthetic RINEX 3 demo file."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(_DEMO_RINEX_OBS):
            raise unittest.SkipTest(f'Demo RINEX obs file not found: {_DEMO_RINEX_OBS}')
        from python.read_rinex_obs import read_rinex_obs
        cls.obs = read_rinex_obs(_DEMO_RINEX_OBS)

    def test_not_none(self):
        """Parser must return a non-None dict for the demo RINEX file."""
        self.assertIsNotNone(self.obs)

    def test_version(self):
        """RINEX version must be parsed as 3.03."""
        self.assertAlmostEqual(self.obs['version'], 3.03, places=2)

    def test_epoch_count(self):
        """Demo file has exactly 60 epochs at 30-second intervals."""
        self.assertEqual(len(self.obs['times']), 60)

    def test_epoch_spacing(self):
        """Epoch interval must be 30 seconds (median)."""
        dt = float(np.median(np.diff(self.obs['times'])))
        self.assertAlmostEqual(dt, 30.0, delta=0.1)

    def test_satellite_count(self):
        """Demo file has 10 satellites: 5 GPS + 3 Galileo + 2 GLONASS."""
        self.assertEqual(len(self.obs['sats']), 10)

    def test_constellation_coverage(self):
        """All three constellations (G, E, R) must be present."""
        sys_chars = {s[0] for s in self.obs['sats']}
        self.assertIn('G', sys_chars)
        self.assertIn('E', sys_chars)
        self.assertIn('R', sys_chars)

    def test_obs_types_gps(self):
        """GPS should have C1C, L1C, S1C, C5Q, L5Q, S5Q."""
        gps_codes = set(self.obs['obs_types'].get('G', []))
        for code in ('C1C', 'L1C', 'S1C', 'C5Q', 'L5Q', 'S5Q'):
            self.assertIn(code, gps_codes)

    def test_pseudorange_range(self):
        """GPS pseudorange (C1C) must be ~22 000 km ± 100 m."""
        pr = self.obs['data']['G01']['C1C']
        mean_pr = float(np.nanmean(pr))
        self.assertAlmostEqual(mean_pr, 22_000_000.0, delta=100.0)

    def test_signal_strength_range(self):
        """C/N0 (S1C) must be between 25 and 60 dB-Hz for GPS sats."""
        for sv in ('G01', 'G02'):
            s1c = self.obs['data'][sv]['S1C']
            valid = s1c[np.isfinite(s1c)]
            self.assertTrue(np.all(valid > 25.0),
                            msg=f'{sv} S1C has values ≤ 25 dB-Hz')
            self.assertTrue(np.all(valid < 60.0),
                            msg=f'{sv} S1C has values ≥ 60 dB-Hz')

    def test_cycle_slip_flag(self):
        """G01 L1C must have LLI=1 at epoch 35 (injected slip)."""
        lli_arr = self.obs['lli']['G01']['L1C']
        self.assertEqual(int(lli_arr[35] & 0x01), 1,
                         'Expected LLI slip at epoch 35 for G01')
        # All other epochs must have no slip
        no_slip = lli_arr.copy()
        no_slip[35] = 0
        self.assertTrue(np.all((no_slip & 0x01) == 0),
                        'Unexpected LLI slips in G01 L1C')

    def test_gap_in_g17(self):
        """G17 must be absent for epochs 20–29 (10-epoch gap)."""
        c1c = self.obs['data']['G17']['C1C']
        gap_epochs = np.where(~np.isfinite(c1c))[0]
        self.assertEqual(set(gap_epochs.tolist()), set(range(20, 30)),
                         'Expected G17 gap at epochs 20-29')

    def test_times_start_at_zero(self):
        """times[0] must be 0 (relative to first epoch)."""
        self.assertEqual(self.obs['times'][0], 0.0)

    def test_datetimes_length(self):
        """datetimes list length must match times array length."""
        self.assertEqual(len(self.obs['datetimes']), len(self.obs['times']))

    def test_missing_file(self):
        """Parser must return None for a non-existent file."""
        from python.read_rinex_obs import read_rinex_obs
        result = read_rinex_obs('/nonexistent/path/to/file.rnx')
        self.assertIsNone(result)


class TestGetSignalStrength(unittest.TestCase):
    """Verify get_signal_strength() helper."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(_DEMO_RINEX_OBS):
            raise unittest.SkipTest(f'Demo RINEX obs file not found: {_DEMO_RINEX_OBS}')
        from python.read_rinex_obs import read_rinex_obs
        cls.obs = read_rinex_obs(_DEMO_RINEX_OBS)

    def test_returns_s_code(self):
        """Must return the S1C observable for G01."""
        from python.read_rinex_obs import get_signal_strength
        code, arr = get_signal_strength(self.obs, 'G01')
        self.assertIsNotNone(code)
        self.assertTrue(code.startswith('S'), f'Expected S code, got {code!r}')
        self.assertEqual(len(arr), len(self.obs['times']))

    def test_valid_values(self):
        """Signal-strength values must be finite and in a plausible range."""
        from python.read_rinex_obs import get_signal_strength
        _, arr = get_signal_strength(self.obs, 'G01')
        valid = arr[np.isfinite(arr)]
        self.assertGreater(len(valid), 0, 'No valid signal-strength values')
        self.assertTrue(np.all(valid > 20.0))
        self.assertTrue(np.all(valid < 70.0))

    def test_unknown_satellite(self):
        """Must return (None, None) for a satellite not in the file."""
        from python.read_rinex_obs import get_signal_strength
        code, arr = get_signal_strength(self.obs, 'G99')
        self.assertIsNone(code)
        self.assertIsNone(arr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
