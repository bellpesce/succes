"""
era5_fetch.py  —  Download ERA5 January 2024 weather data from Copernicus CDS
------------------------------------------------------------------------------
Place this file anywhere and run it from the project root OR from the data/ folder.
It always saves downloads to the data/ subfolder next to era5_to_profiles.py.

Requirements
------------
    pip install cdsapi

Credentials — three options (use whichever is easiest):
-------------------------------------------------------
Option A (recommended for Windows):
    Pass key directly on the command line — no config file needed:
        python era5_fetch.py --key YOUR_TOKEN_HERE

Option B: environment variables (set once in your terminal session):
    set CDSAPI_URL=https://cds.climate.copernicus.eu/api
    set CDSAPI_KEY=YOUR_TOKEN_HERE
    python era5_fetch.py

Option C: config file at C:\\Users\\YourName\\.cdsapirc
    Create the file with Notepad (no .txt extension):
        url: https://cds.climate.copernicus.eu/api
        key: YOUR_TOKEN_HERE

Get your token: log in at cds.climate.copernicus.eu → top-right menu → API tokens.
Accept the ERA5 licence on the dataset page before first download.

Usage
-----
    python era5_fetch.py --key YOUR_TOKEN_HERE        # recommended
    python era5_fetch.py --key YOUR_TOKEN_HERE --test  # quick connectivity test
    python era5_fetch.py --year 2023                   # different year

What is downloaded (into the data/ folder)
-------------------------------------------
    era5_202401_wind.nc    ~8 MB   100m u/v wind components
    era5_202401_solar.nc   ~4 MB   surface solar radiation downwelling
    era5_202401_temp.nc    ~4 MB   2m temperature

Total: ~16 MB.  CDS queue time: 2–10 minutes.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# ── Locate the data/ directory reliably regardless of where script is run ─────
# Strategy: look for era5_to_profiles.py as the anchor.
# Search: next to this script, then ../data/, then ./data/
def _find_data_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,                        # script is already in data/
        script_dir / "data",               # script in project root
        script_dir.parent / "data",        # script in examples/
    ]
    for c in candidates:
        if (c / "era5_to_profiles.py").exists():
            return c
    # Fallback: create data/ next to script
    d = script_dir / "data" if not script_dir.name == "data" else script_dir
    d.mkdir(parents=True, exist_ok=True)
    return d

DATA_DIR = _find_data_dir()

# ── Bounding box covering all 15 SUCCES regions ───────────────────────────────
AREA = [61.0, -9.5, 42.0, 20.5]   # North, West, South, East

BASE_REQUEST = {
    "product_type":    ["reanalysis"],
    "data_format":     "netcdf",
    "download_format": "unarchived",
    "area": AREA,
    "grid": [0.25, 0.25],
}


def make_time_list():
    return [f"{h:02d}:00" for h in range(24)]


def make_day_list(year: int, month: int):
    import calendar
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def make_client(key: str | None):
    """Create a cdsapi.Client, injecting key if provided."""
    import cdsapi
    if key:
        # Pass credentials directly — no config file needed
        return cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=key,
            quiet=False,
        )
    # Fall back to environment variables or ~/.cdsapirc
    return cdsapi.Client(quiet=False)


def _download(variable_label: str, variables: list[str], suffix: str,
              year: int, month: int, test: bool, key: str | None) -> Path:
    out_path = DATA_DIR / f"era5_{year}{month:02d}_{suffix}.nc"
    if out_path.exists():
        mb = out_path.stat().st_size / 1e6
        print(f"  [{variable_label:5s}] Already exists: {out_path.name} ({mb:.1f} MB) — skipping.")
        return out_path

    client = make_client(key)
    request = {
        **BASE_REQUEST,
        "variable": variables,
        "year":     [str(year)],
        "month":    [f"{month:02d}"],
        "day":      ["01"] if test else make_day_list(year, month),
        "time":     ["00:00", "12:00"] if test else make_time_list(),
    }
    print(f"  [{variable_label:5s}] Requesting from CDS…  (may take 2-10 min in queue)")
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    mb = out_path.stat().st_size / 1e6
    print(f"  [{variable_label:5s}] Saved → {out_path.name}  ({mb:.1f} MB)")
    return out_path


def download_wind(year, month, test, key):
    return _download("wind", [
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
    ], "wind", year, month, test, key)


def download_solar(year, month, test, key):
    return _download("solar", [
        "surface_solar_radiation_downwards",
    ], "solar", year, month, test, key)


def download_temp(year, month, test, key):
    return _download("temp", [
        "2m_temperature",
    ], "temp", year, month, test, key)


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 weather data for SUCCES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python era5_fetch.py --key abc123def456          # pass token directly
  python era5_fetch.py --key abc123def456 --test   # quick test (1 day)
  python era5_fetch.py                             # use env var or config file
        """,
    )
    parser.add_argument("--year",  type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--test",  action="store_true",
                        help="Download 1 day / 2 time steps only (connectivity test)")
    parser.add_argument("--key",   type=str, default=None,
                        help="CDS API personal access token (bypasses config file)")
    args = parser.parse_args()

    # Also accept key from environment
    key = args.key or os.environ.get("CDSAPI_KEY")

    print(f"\nSUCCES ERA5 Fetcher — {args.year}-{args.month:02d}"
          + (" [TEST MODE]" if args.test else ""))
    print(f"Output directory: {DATA_DIR.resolve()}")
    print(f"Bounding box:     N={AREA[0]} S={AREA[2]} W={AREA[1]} E={AREA[3]}")

    if key:
        print(f"Credentials:      --key argument (token: …{key[-6:]})")
    elif os.environ.get("CDSAPI_URL"):
        print("Credentials:      CDSAPI_URL / CDSAPI_KEY environment variables")
    else:
        rc = os.path.expanduser("~/.cdsapirc")
        if Path(rc).exists():
            print(f"Credentials:      config file at {rc}")
        else:
            print(f"\n  ERROR: No credentials found.")
            print(f"  Easiest fix — pass your token directly:")
            print(f"    python era5_fetch.py --key YOUR_TOKEN_HERE")
            print(f"\n  Get your token: log in at cds.climate.copernicus.eu")
            print(f"  → top-right profile menu → API tokens → copy the token")
            print(f"\n  Alternatively, create: {rc}")
            print(f"  with contents:")
            print(f"    url: https://cds.climate.copernicus.eu/api")
            print(f"    key: YOUR_TOKEN_HERE")
            sys.exit(1)

    if args.test:
        print("Mode:             TEST (1 day, 2 time steps — fast)\n")
    else:
        print("Mode:             Full month (~744 hours, ~16 MB total)\n")

    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed.")
        print("  Run: pip install cdsapi")
        sys.exit(1)

    try:
        wind_path  = download_wind(args.year, args.month, args.test, key)
        solar_path = download_solar(args.year, args.month, args.test, key)
        temp_path  = download_temp(args.year, args.month, args.test, key)

    except Exception as e:
        err = str(e)
        print(f"\n  ERROR: {err}\n")
        if "401" in err or "authoriz" in err.lower() or "key" in err.lower():
            print("  → Invalid or missing API key.")
            print("    Get your token at: cds.climate.copernicus.eu → profile → API tokens")
            print("    Then run: python era5_fetch.py --key YOUR_TOKEN_HERE")
        elif "licence" in err.lower() or "terms" in err.lower():
            print("  → You need to accept the ERA5 licence.")
            print("    Visit: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels")
            print("    Scroll down and click 'Accept terms'")
        elif "timeout" in err.lower() or "connect" in err.lower():
            print("  → Network error. CDS can be slow — try again in a few minutes.")
        else:
            print("  → Check your internet connection and CDS service status at:")
            print("    https://cds.climate.copernicus.eu")
        sys.exit(1)

    print(f"\n  Downloads complete:")
    print(f"    {wind_path.name}   ({wind_path.stat().st_size/1e6:.1f} MB)")
    print(f"    {solar_path.name}  ({solar_path.stat().st_size/1e6:.1f} MB)")
    print(f"    {temp_path.name}   ({temp_path.stat().st_size/1e6:.1f} MB)")
    print(f"\n  Next step:")
    print(f"    python data/era5_to_profiles.py")


if __name__ == "__main__":
    main()
