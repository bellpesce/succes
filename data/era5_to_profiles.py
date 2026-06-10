"""
data/era5_to_profiles.py  —  Convert ERA5 netCDF to SUCCES demand profiles
---------------------------------------------------------------------------
Run after era5_fetch.py.  No CDS account needed — processes local files only.

Usage
-----
    python data/era5_to_profiles.py                  # Jan 2024
    python data/era5_to_profiles.py --year 2023      # different year
    python data/era5_to_profiles.py --test            # uses synthetic data

Outputs
-------
    data/era5_jan2024_profiles.npz   — numpy archive, <1 MB
        Arrays (all shape T × len(REGIONS) unless noted):
            wind_mw    (T, R)  — hourly wind generation (MW) by region
            solar_mw   (T, R)  — hourly solar generation (MW) by region
            temp_scale (T, R)  — hourly demand temperature scaling factor
            wind_cf    (T, R)  — wind capacity factor [0,1]
            solar_cf   (T, R)  — solar capacity factor [0,1]
            regions    (R,)    — region name strings
            hours      int     — T (number of hours)
            year       int
            month      int

Physics
-------
Wind CF:
    IEC class II power curve.  100m wind speed u/v → speed → CF.
    Cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s.
    Exponent 2.8 (slightly sub-cubic, matches real fleet averaging).

Solar CF:
    ERA5 ssrd is accumulated J/m² from 00:00 UTC.
    Differenced to get hourly W/m².
    CF = irradiance × panel_efficiency × performance_ratio / STC_irradiance
       = irradiance × 0.20 × 0.80 / 1000  →  max ≈ 16% (January realistic).

Temperature demand scaling:
    Heating degree hours above 15°C base temperature.
    scale = 1 + 0.006 × max(15 - T_celsius, 0)
    At -10°C: scale = 1.15 (+15% demand).
    At +15°C: scale = 1.00 (no effect).
    This replaces the synthetic TEMP_SCALE_JAN2024 table in scenarios.

Representative grid points:
    One ERA5 0.25° grid point per region, chosen for the dominant
    generation technology (offshore-biased for wind-heavy regions).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent

# ── Dual ERA5 extraction points per region ────────────────────────────────────
# Each region has two wind extraction points:
#   OFFSHORE_COORDS: coastal or offshore point — high Jan CF (0.40-0.70)
#                    used for offshore/nearshore installed capacity
#   INLAND_COORDS:   best onshore cluster near population centre
#                    used for onshore installed capacity
#
# Temperature and solar use the inland point (population-representative).
#
# The blended wind profile:
#   wind_mw[t] = offshore_cf[t] * offshore_MW + inland_cf[t] * inland_MW
#
# This correctly mixes the high-CF offshore resource with the lower-CF
# inland fleet, eliminating the single-point offshore bias that previously
# caused 90%+ of GB/ES/FR hours to have zero or negative prices.
#
# For regions with no offshore (PL, IT, AT, CH, CZ, HU):
# offshore_MW = 0, offshore_coord = inland_coord (unused but kept for consistency).
#
# Sources: WindEurope 2024, ENTSO-E SOAF 2023, national TSO reports.

OFFSHORE_COORDS = {
    "AT": (47.50, 14.50),  # no offshore — same as inland
    "BE": (51.50,  2.50),  # Belgian coast, Princess Elisabeth zone
    "CH": (47.00,  8.00),  # no offshore
    "CZ": (49.75, 15.75),  # no offshore
    "DE": (54.00,  8.00),  # German Bight — North Sea offshore
    "DK": (55.75,  8.25),  # W. Jutland — offshore-representative
    "ES": (43.75, -7.75),  # Galicia Atlantic coast
    "FR": (47.00, -2.00),  # Loire Atlantique / Bretagne
    "GB": (53.50, -3.00),  # Irish Sea — Hornsea, Walney, Dogger corridor
    "HU": (47.25, 19.00),  # no offshore
    "IT": (38.50, 15.50),  # Strait of Messina — no significant offshore yet
    "NL": (52.25,  4.75),  # Holland coast — Borssele, Hollandse Kust
    "NO": (60.50,  5.25),  # Vestlandet — Hywind Tampen
    "PL": (54.75, 18.50),  # Baltic coast — no significant offshore yet
    "SE": (57.00, 11.50),  # Kattegat coast — some nearshore
}

INLAND_COORDS = {
    "AT": (48.00, 16.50),  # Marchfeld / Lower Austria — main onshore cluster
    "BE": (50.50,  4.75),  # Central Belgium — Wallonia cluster
    "CH": (47.00,  8.00),  # Swiss Mittelland
    "CZ": (49.75, 15.75),  # Central Bohemia
    "DE": (51.50, 10.00),  # Central Germany — Thuringia/Saxony onshore cluster
    "DK": (56.25,  9.50),  # Central Jutland — inland onshore
    "ES": (41.00, -4.00),  # Castilla y León — largest onshore cluster
    "FR": (44.00,  2.50),  # Southern France — Occitanie/Massif Central
    "GB": (55.00, -3.50),  # Scottish Southern Uplands — largest onshore cluster
    "HU": (47.25, 19.00),  # Central Hungary
    "IT": (41.00, 15.50),  # Puglia / Campania — main onshore
    "NL": (52.50,  5.50),  # Central Netherlands — Flevoland/inland
    "NO": (61.50,  9.50),  # Oppland / Fjell onshore
    "PL": (52.00, 19.50),  # Central Poland — main cluster
    "SE": (58.00, 15.00),  # Götaland inland — Småland/Östergötland cluster
}

REGIONS = sorted(OFFSHORE_COORDS.keys())

# Legacy single-point for temperature and solar (inland = population centre)
REGION_COORDS = INLAND_COORDS   # used by process_temp, process_solar

# ── Installed capacity (MW) for demand subtraction ───────────────────────────
# Source: ENTSO-E SOAF 2023, IRENA, national TSO data (approximate 2024 values)

# ── Installed capacity split: offshore vs inland (MW, Jan 2024) ───────────────
# Sources: WindEurope 2024 Offshore Statistics, ENTSO-E SOAF 2023.
WIND_OFFSHORE_MW = {
    "DE":  8_000,  "DK":  2_800,  "GB": 14_500,  "NL":  3_500,
    "BE":  2_300,  "FR":  1_500,  "NO":    300,  "SE":  3_600,
    "ES": 10_000,  # Atlantic coast cluster (treated as coastal/semi-offshore)
    "PL":      0,  "IT":      0,  "AT":      0,
    "CH":      0,  "CZ":      0,  "HU":      0,
}
WIND_INLAND_MW = {
    "DE": 60_000,  "DK":  3_700,  "GB": 13_500,  "NL":  3_000,
    "BE":  3_200,  "FR": 20_500,  "NO":  3_700,  "SE":  8_400,
    "ES": 20_000,  "PL":  8_000,  "IT": 12_000,  "AT":  3_500,
    "CH":    100,  "CZ":    350,  "HU":    250,
}
# Total installed = offshore + inland (used for WindPlant.max_cap)
WIND_INSTALLED = {
    r: WIND_OFFSHORE_MW.get(r, 0) + WIND_INLAND_MW.get(r, 0)
    for r in REGIONS
}
SOLAR_INSTALLED = {
    "DE": 81_000,  # huge installed, very low Jan CF
    "IT": 24_000,
    "ES": 20_000,
    "FR": 19_000,
    "NL":  9_000,
    "GB": 15_000,
    "PL":  9_000,
    "BE":  7_500,
    "CZ":  3_000,
    "AT":  4_000,
    "CH":  4_000,
    "SE":  2_000,
    "DK":  3_500,
    "NO":    100,
    "HU":  4_500,
}

# ── Physics constants ─────────────────────────────────────────────────────────
WIND_CUT_IN     =  3.0   # m/s
WIND_RATED      = 12.0   # m/s
WIND_CUT_OUT    = 25.0   # m/s
WIND_EXPONENT   =  2.8   # sub-cubic, matches fleet-averaged power curves

PANEL_EFF       = 0.20   # monocrystalline Si typical
PANEL_PR        = 0.80   # performance ratio
STC_W_M2        = 1000.0

TEMP_BASE_C     = 15.0   # heating degree base temperature (°C)
TEMP_ALPHA      = 0.006  # demand fraction per degree below base


# ── Conversion functions ──────────────────────────────────────────────────────

def wind_cf_iec2(ws_ms: np.ndarray) -> np.ndarray:
    """100m wind speed (m/s) → capacity factor [0, 1]."""
    ws = np.asarray(ws_ms, dtype=float)
    cf = np.zeros_like(ws)
    ramp = (ws >= WIND_CUT_IN) & (ws < WIND_RATED)
    cf[ramp] = np.clip(((ws[ramp] - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)) ** WIND_EXPONENT, 0.0, 1.0)
    cf[(ws >= WIND_RATED) & (ws <= WIND_CUT_OUT)] = 1.0
    return cf


def solar_cf_from_ssrd(ssrd_j_m2: np.ndarray) -> np.ndarray:
    """
    Accumulated SSRD (J/m², length T+1) → hourly CF (length T).
    ERA5 ssrd is accumulated from 00:00 UTC; diff gives hourly energy.
    """
    hourly_w = np.diff(np.asarray(ssrd_j_m2, dtype=float)) / 3600.0
    hourly_w = np.maximum(hourly_w, 0.0)
    return np.clip(hourly_w * PANEL_EFF * PANEL_PR / STC_W_M2, 0.0, 1.0)


def temp_demand_scale(t2m_kelvin: np.ndarray) -> np.ndarray:
    """2m temperature (K) → demand scaling factor. Values typically 0.88–1.15."""
    t_c = np.asarray(t2m_kelvin, dtype=float) - 273.15
    hdd = np.maximum(TEMP_BASE_C - t_c, 0.0)
    return 1.0 + TEMP_ALPHA * hdd


def _open_nc(path: Path):
    """
    Open a netCDF file with xarray, trying engines in order.
    Gives a clear install message if none work.
    """
    import xarray as xr
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(str(path), engine=engine)
        except Exception:
            continue
    print("\n  ERROR: Cannot open the netCDF file.")
    print("  Install the missing backend:")
    print("    pip install netCDF4")
    print("  or:")
    print("    pip install h5netcdf")
    print(f"\n  File: {path}")
    import sys; sys.exit(1)


def extract_nearest_point(ds, lat: float, lon: float, var: str) -> np.ndarray:
    """
    Extract time series from xarray Dataset at the nearest grid point to (lat, lon).
    Handles both CF-convention lat/lon names and ERA5-style latitude/longitude.
    Returns (T,) numpy array.
    """
    lat_dim = next((d for d in ds.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in ds.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Cannot find lat/lon dims in dataset. Dims: {list(ds.dims)}")

    sel_kwargs = {lat_dim: lat, lon_dim: lon}
    point = ds[var].sel(**sel_kwargs, method="nearest")

    # Drop any extra singleton dims (e.g. expver)
    for dim in list(point.dims):
        if dim not in ("time", "valid_time"):
            point = point.isel({dim: 0})

    return point.values.astype(float)
    """
    Extract time series from xarray Dataset at the nearest grid point to (lat, lon).
    Handles both CF-convention lat/lon names and ERA5-style latitude/longitude.
    Returns (T,) numpy array.
    """
    # Find dimension names
    lat_dim = next((d for d in ds.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in ds.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Cannot find lat/lon dims in dataset. Dims: {list(ds.dims)}")

    # Select nearest
    sel_kwargs = {lat_dim: lat, lon_dim: lon}
    point = ds[var].sel(**sel_kwargs, method="nearest")

    # Drop any extra singleton dims (e.g. expver)
    for dim in list(point.dims):
        if dim not in ("time", "valid_time"):
            point = point.isel({dim: 0})

    return point.values.astype(float)


def process_wind(nc_path: Path, year: int, month: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process wind netCDF → (wind_cf_blended, wind_mw_blended) arrays of shape (T, R).

    For each region extracts two ERA5 points:
      - OFFSHORE_COORDS: coastal/offshore point — high Jan CF (0.40-0.70)
      - INLAND_COORDS:   best onshore cluster — lower Jan CF (0.15-0.35)

    Blends them by installed capacity:
      wind_mw[t, r] = offshore_cf[t] * offshore_MW[r]
                    + inland_cf[t]   * inland_MW[r]

    The blended wind_cf = wind_mw / total_installed is the fleet-weighted
    average capacity factor, correctly mixing high-CF offshore with lower-CF
    inland across all hours.
    """
    import xarray as xr
    ds = _open_nc(nc_path)
    print(f"    Wind variables: {list(ds.data_vars)}")

    u_var = next((v for v in ds.data_vars if "u100" in v.lower() or "100u" in v.lower()), None)
    v_var = next((v for v in ds.data_vars if "v100" in v.lower() or "100v" in v.lower()), None)
    if u_var is None or v_var is None:
        raise ValueError(f"Cannot find u100/v100 in {list(ds.data_vars)}")

    T = len(ds["time"]) if "time" in ds.dims else len(ds["valid_time"])
    R = len(REGIONS)
    wind_cf = np.zeros((T, R))
    wind_mw = np.zeros((T, R))

    for ri, r in enumerate(REGIONS):
        off_MW = float(WIND_OFFSHORE_MW.get(r, 0))
        in_MW  = float(WIND_INLAND_MW.get(r, 0))
        total  = off_MW + in_MW

        # Always extract inland point
        ilat, ilon = INLAND_COORDS[r]
        iu = extract_nearest_point(ds, ilat, ilon, u_var)
        iv = extract_nearest_point(ds, ilat, ilon, v_var)
        inland_cf = wind_cf_iec2(np.sqrt(iu**2 + iv**2))

        # Extract offshore point only if capacity > 0
        if off_MW > 0.0:
            olat, olon = OFFSHORE_COORDS[r]
            ou = extract_nearest_point(ds, olat, olon, u_var)
            ov = extract_nearest_point(ds, olat, olon, v_var)
            offshore_cf = wind_cf_iec2(np.sqrt(ou**2 + ov**2))
        else:
            offshore_cf = np.zeros(T)

        # Blended wind output (MW) = offshore contribution + inland contribution
        blended_mw = offshore_cf * off_MW + inland_cf * in_MW

        wind_mw[:, ri] = blended_mw
        wind_cf[:, ri] = blended_mw / total if total > 0 else np.zeros(T)

    ds.close()
    return wind_cf, wind_mw


def process_solar(nc_path: Path, year: int, month: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process solar netCDF → (solar_cf, solar_mw) arrays of shape (T, R).
    ERA5 ssrd is accumulated → we need to diff per day.
    """
    import xarray as xr
    ds = _open_nc(nc_path)

    ssrd_var = next((v for v in ds.data_vars if "ssrd" in v.lower()
                     or "solar" in v.lower() or "radiation" in v.lower()), None)
    if ssrd_var is None:
        raise ValueError(f"Cannot find ssrd in {list(ds.data_vars)}")

    T = len(ds["time"]) if "time" in ds.dims else len(ds["valid_time"])
    R = len(REGIONS)
    solar_cf = np.zeros((T, R))
    solar_mw = np.zeros((T, R))

    for ri, r in enumerate(REGIONS):
        lat, lon = REGION_COORDS[r]
        ssrd_raw = extract_nearest_point(ds, lat, lon, ssrd_var)

        # ERA5 ssrd accumulates within each day, resetting at 00:00 UTC.
        # We reconstruct hourly irradiance by differencing within each day.
        hourly_w = np.zeros(T)
        for d in range(T // 24):
            day_slice = ssrd_raw[d*24 : (d+1)*24]
            # Day resets at midnight: first hour = day_slice[0] J/m² total
            # hourly irradiance = diff within the day, first hour is already hourly
            day_hourly = np.empty(24)
            day_hourly[0] = day_slice[0]
            for h in range(1, 24):
                diff = day_slice[h] - day_slice[h-1]
                # If diff is negative, the accumulator reset → that hour = slice[h]
                day_hourly[h] = diff if diff >= 0 else day_slice[h]
            hourly_w[d*24:(d+1)*24] = day_hourly / 3600.0  # J/m² → W/m²

        # Handle any remaining hours (e.g. months with 31 days: 744 = 31*24)
        hourly_w = np.maximum(hourly_w, 0.0)
        cf = np.clip(hourly_w * PANEL_EFF * PANEL_PR / STC_W_M2, 0.0, 1.0)
        solar_cf[:, ri] = cf
        solar_mw[:, ri] = cf * SOLAR_INSTALLED.get(r, 0)

    ds.close()
    return solar_cf, solar_mw


def process_temp(nc_path: Path, year: int, month: int) -> np.ndarray:
    """
    Process temperature netCDF → temp_scale array of shape (T, R).
    """
    import xarray as xr
    ds = _open_nc(nc_path)

    t2m_var = next((v for v in ds.data_vars if "2m" in v.lower() or "t2m" in v.lower()
                    or "temperature" in v.lower()), None)
    if t2m_var is None:
        raise ValueError(f"Cannot find t2m in {list(ds.data_vars)}")

    T = len(ds["time"]) if "time" in ds.dims else len(ds["valid_time"])
    R = len(REGIONS)
    temp_scale = np.zeros((T, R))

    for ri, r in enumerate(REGIONS):
        lat, lon = REGION_COORDS[r]
        t2m = extract_nearest_point(ds, lat, lon, t2m_var)
        temp_scale[:, ri] = temp_demand_scale(t2m)

    ds.close()
    return temp_scale


def make_synthetic_profiles(year: int, month: int) -> tuple:
    """
    Generate synthetic profiles for testing without CDS data.
    Mimics ERA5 Jan 2024 statistics based on climatology.
    """
    import calendar
    T = calendar.monthrange(year, month)[1] * 24
    R = len(REGIONS)
    rng = np.random.default_rng(42)

    BASE_WS_OFFSHORE = {"DK":9.0,"GB":8.5,"NL":8.0,"DE":8.0,"BE":8.0,"FR":7.5,
                        "NO":7.0,"SE":7.5,"ES":7.0,"AT":4.0,"CH":3.5,"CZ":4.5,
                        "HU":3.8,"PL":5.0,"IT":4.5}
    BASE_WS_INLAND   = {"DE":5.5,"DK":5.5,"GB":5.0,"NL":5.0,"FR":5.0,"BE":5.5,
                        "ES":6.0,"NO":5.5,"SE":5.5,"PL":5.0,"IT":4.5,
                        "AT":4.0,"CH":3.5,"CZ":4.5,"HU":3.8}
    BASE_T  = {"FR":279,"DE":276,"GB":280,"ES":285,"IT":282,"NO":273,
               "SE":271,"DK":278,"NL":279,"BE":279,"AT":274,"CH":273,
               "CZ":274,"PL":273,"HU":275}
    LAT_SOLAR = {"ES":0.35,"IT":0.30,"FR":0.20,"GB":0.12,"DE":0.12,
                 "NL":0.12,"BE":0.12,"DK":0.08,"SE":0.06,"NO":0.04}

    wind_cf    = np.zeros((T, R))
    wind_mw    = np.zeros((T, R))
    solar_cf   = np.zeros((T, R))
    solar_mw   = np.zeros((T, R))
    temp_scale = np.zeros((T, R))

    for ri, r in enumerate(REGIONS):
        off_MW = float(WIND_OFFSHORE_MW.get(r, 0))
        in_MW  = float(WIND_INLAND_MW.get(r, 0))
        total  = off_MW + in_MW

        def _synth_cf(base_ws):
            ws_noise = np.cumsum(rng.normal(0, 0.8, T)) * 0.05 + rng.normal(0, 1.5, T)
            ws = np.maximum(0, base_ws + ws_noise)
            smooth = np.zeros(T); smooth[0] = ws[0]
            for t in range(1, T): smooth[t] = 0.85 * smooth[t-1] + 0.15 * ws[t]
            return wind_cf_iec2(smooth)

        cf_off = _synth_cf(BASE_WS_OFFSHORE.get(r, 5.0)) if off_MW > 0 else np.zeros(T)
        cf_in  = _synth_cf(BASE_WS_INLAND.get(r, 5.0))

        blended_mw = cf_off * off_MW + cf_in * in_MW
        wind_mw[:, ri] = blended_mw
        wind_cf[:, ri] = blended_mw / total if total > 0 else np.zeros(T)

        lf = LAT_SOLAR.get(r, 0.10)
        for t in range(T):
            h = t % 24
            if 8 <= h <= 16:
                irr = lf * 900 * np.sin(np.pi * (h-8) / 8) * (0.8 + 0.2 * rng.random())
                cf_s = np.clip(irr * PANEL_EFF * PANEL_PR / STC_W_M2, 0.0, 1.0)
                solar_cf[t, ri] = cf_s
                solar_mw[t, ri] = cf_s * SOLAR_INSTALLED.get(r, 0)

        base_t = float(BASE_T.get(r, 276))
        t_series = np.array([
            base_t + (-3.0 if 7 <= t//24 <= 14 else 2.0) + rng.normal(0, 1.5)
            for t in range(T)
        ])
        temp_scale[:, ri] = temp_demand_scale(t_series)

    return wind_cf, wind_mw, solar_cf, solar_mw, temp_scale


def main():
    parser = argparse.ArgumentParser(description="Convert ERA5 netCDF to SUCCES profiles")
    parser.add_argument("--year",  type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--test",  action="store_true",
                        help="Use synthetic data instead of ERA5 files")
    args = parser.parse_args()

    tag      = f"{args.year}{args.month:02d}"
    out_path = DATA_DIR / f"era5_{tag}_profiles.npz"

    print(f"\nSUCCES ERA5 Profile Builder — {args.year}-{args.month:02d}")
    print(f"Output: {out_path}")

    # Pre-flight: check that at least one netCDF backend is available
    _nc_backend = None
    for _pkg in ("netCDF4", "h5netcdf", "scipy"):
        try:
            __import__(_pkg.replace("-", "_").lower().replace("netcdf4","netCDF4"))
            _nc_backend = _pkg; break
        except ImportError:
            pass
    if _nc_backend is None:
        print("\n  ERROR: No netCDF reader installed.")
        print("  Run one of:")
        print("    pip install netCDF4")
        print("    pip install h5netcdf")
        sys.exit(1)
    print(f"  netCDF backend: {_nc_backend}")

    if args.test:
        print("TEST MODE: using synthetic profiles (no ERA5 files needed)")
        wind_cf, wind_mw, solar_cf, solar_mw, temp_scale = \
            make_synthetic_profiles(args.year, args.month)
    else:
        wind_nc  = DATA_DIR / f"era5_{tag}_wind.nc"
        solar_nc = DATA_DIR / f"era5_{tag}_solar.nc"
        temp_nc  = DATA_DIR / f"era5_{tag}_temp.nc"
        for f in [wind_nc, solar_nc, temp_nc]:
            if not f.exists():
                print(f"ERROR: {f.name} not found. Run era5_fetch.py first.")
                sys.exit(1)

        print("Processing wind ...")
        wind_cf, wind_mw = process_wind(wind_nc, args.year, args.month)
        print("Processing solar ...")
        solar_cf, solar_mw = process_solar(solar_nc, args.year, args.month)
        print("Processing temperature ...")
        temp_scale = process_temp(temp_nc, args.year, args.month)

    T = wind_cf.shape[0]

    # ── Validation ────────────────────────────────────────────────────────────
    print(f"\nProfile validation ({T}h × {len(REGIONS)} regions):")
    print(f"  {'R':4}  {'Wind_CF':>8}  {'Wind_MW':>9}  {'Solar_CF':>9}  {'Temp_sc':>8}")
    for ri, r in enumerate(REGIONS):
        print(f"  {r:4}  {wind_cf[:,ri].mean():8.3f}  {wind_mw[:,ri].mean():9.0f}  "
              f"{solar_cf[:,ri].mean():9.4f}  {temp_scale[:,ri].mean():8.4f}")

    # Sanity checks
    assert np.all(wind_cf  >= 0) and np.all(wind_cf  <= 1), "Wind CF out of [0,1]"
    assert np.all(solar_cf >= 0) and np.all(solar_cf <= 1), "Solar CF out of [0,1]"
    assert np.all(temp_scale >= 0.8) and np.all(temp_scale <= 1.3), \
        f"Temp scale out of range: min={temp_scale.min():.3f} max={temp_scale.max():.3f}"
    assert np.all(solar_cf[0::24] == 0), "Solar should be zero at midnight"
    print("\n  ✓ All sanity checks passed")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        str(out_path),
        wind_cf    = wind_cf,      # (T, R)
        wind_mw    = wind_mw,      # (T, R)
        solar_cf   = solar_cf,     # (T, R)
        solar_mw   = solar_mw,     # (T, R)
        temp_scale = temp_scale,   # (T, R)
        regions    = np.array(REGIONS),
        hours      = np.int64(T),
        year       = np.int64(args.year),
        month      = np.int64(args.month),
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved → {out_path.name}  ({size_kb:.0f} KB)")
    print(f"\nNext step: python examples/europe_nordic_jan2024.py")
    print(f"  The scenario will automatically use ERA5 profiles from {out_path.name}")


if __name__ == "__main__":
    main()
