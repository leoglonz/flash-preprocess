r"""Download USGS 15-minute instantaneous discharge for gages in an events CSV.

Fetches raw instantaneous-values discharge (parameter 00060) from the USGS
NWIS API for every gage in `--events` (STAID column), converts timestamps
from local (mixed-offset) time to UTC, and resamples to a fixed 15-minute
UTC grid. Output is a long-format CSV:

    STAID, site_name, datetime, discharge_cfs, latitude, longitude

This is a script version of discharge_data.ipynb / discharge.ipynb, driven
off the events CSV (which carries the gage list we actually need obs for)
instead of a separate huc8_events_and_gages.csv.

Usage
-----
    python engine/streamflow/usgs/download_discharge.py \
        --events /gpfs/leoglonz/sub_hourly/data/upper_neuse_usgs/events.csv \
        --start 2021-01-01 --end 2025-12-31 \
        --output /gpfs/leoglonz/sub_hourly/data/upper_neuse_usgs/usgs_discharge.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger('DownloadDischarge')

NWIS_IV_URL = 'https://waterservices.usgs.gov/nwis/iv/'


def load_gage_ids(events_csv: Path, staid_col: str) -> list[str]:
    """Read unique, zero-padded gage STAIDs from an events/gages CSV."""
    df = pd.read_csv(events_csv, dtype={staid_col: str})
    return (
        df[staid_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace('.0', '', regex=False)
        .str.zfill(8)
        .drop_duplicates()
        .tolist()
    )


def fetch_discharge(
    site: str,
    start_date: str,
    end_date: str,
    parameter_code: str,
) -> pd.DataFrame:
    """Fetch raw instantaneous discharge for one gage from the NWIS IV service."""
    params = {
        'format': 'json',
        'sites': site,
        'parameterCd': parameter_code,
        'startDT': start_date,
        'endDT': end_date,
        'siteStatus': 'all',
    }
    r = requests.get(NWIS_IV_URL, params=params, timeout=60)
    r.raise_for_status()

    series = r.json()['value'].get('timeSeries', [])
    if not series:
        return pd.DataFrame()

    rows = []
    for ts in series:
        info = ts['sourceInfo']
        site_no = info['siteCode'][0]['value']
        site_name = info['siteName']
        lat = info['geoLocation']['geogLocation']['latitude']
        lon = info['geoLocation']['geogLocation']['longitude']

        for obs in ts['values'][0].get('value', []):
            val = obs.get('value')
            rows.append(
                {
                    'STAID': site_no,
                    'site_name': site_name,
                    'datetime': obs['dateTime'],
                    'discharge_cfs': float(val) if val not in (None, '') else None,
                    'latitude': lat,
                    'longitude': lon,
                },
            )

    return pd.DataFrame(rows)


def download_all(
    gage_ids: list[str],
    start_date: str,
    end_date: str,
    parameter_code: str,
) -> pd.DataFrame:
    """Download and concatenate raw discharge for all gages."""
    all_data = []
    empty, failed = [], []

    for gage in tqdm(gage_ids, desc='Downloading gages'):
        try:
            df = fetch_discharge(gage, start_date, end_date, parameter_code)
        except Exception as e:  # noqa: BLE001 - report and continue past per-gage failures
            failed.append((gage, str(e)))
            log.warning('%s: %s', gage, e)
            continue

        if df.empty:
            empty.append(gage)
        else:
            all_data.append(df)

    log.info(
        'Successful gages: %d | Empty: %d | Failed: %d',
        len(all_data),
        len(empty),
        len(failed),
    )
    if empty:
        log.info('Empty gages: %s', empty)
    if failed:
        log.info('Failed gages: %s', failed)

    if not all_data:
        raise RuntimeError('No discharge data retrieved for any gage.')

    discharge = pd.concat(all_data, ignore_index=True)
    # Sort by a parsed helper column but keep the original mixed-offset
    # datetime strings intact; to_utc_15min() does the real UTC conversion.
    sort_key = pd.to_datetime(discharge['datetime'], utc=True)
    discharge = (
        discharge.assign(_sort_key=sort_key)
        .sort_values(['STAID', '_sort_key'])
        .drop(columns='_sort_key')
    )
    return discharge.reset_index(drop=True)


def to_utc_15min(
    discharge: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Convert mixed-offset local timestamps to UTC and resample to a 15-min grid."""
    discharge = discharge.copy()
    discharge['datetime'] = pd.to_datetime(
        discharge['datetime'],
        errors='coerce',
        utc=True,
    )
    discharge = discharge.dropna(subset=['datetime'])
    discharge['datetime'] = discharge['datetime'].dt.tz_localize(None)

    study_start = pd.Timestamp(start_date)
    study_end = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=45)
    discharge = discharge[
        (discharge['datetime'] >= study_start) & (discharge['datetime'] <= study_end)
    ].copy()

    resampled = (
        discharge.set_index('datetime')
        .groupby('STAID')
        .resample('15min')[['discharge_cfs', 'latitude', 'longitude']]
        .max()
        .reset_index()
    )

    site_names = discharge[['STAID', 'site_name']].drop_duplicates()
    resampled = resampled.merge(site_names, on='STAID', how='left')

    return resampled[
        ['STAID', 'site_name', 'datetime', 'discharge_cfs', 'latitude', 'longitude']
    ].sort_values(['STAID', 'datetime'])


def main():
    """Parse CLI args and run the discharge download + resample pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--events',
        type=Path,
        required=True,
        help='Events/gages CSV with a STAID column',
    )
    parser.add_argument(
        '--staid-col',
        default='STAID',
        help='STAID column name (default: %(default)s)',
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Study period start date, e.g. 2021-01-01',
    )
    parser.add_argument(
        '--end',
        required=True,
        help='Study period end date, e.g. 2025-12-31',
    )
    parser.add_argument(
        '--parameter-code',
        default='00060',
        help='NWIS parameter code (default: %(default)s)',
    )
    parser.add_argument(
        '--raw-cache',
        type=Path,
        default=None,
        help='Optional path to cache raw (pre-UTC) download, reused if it already exists',
    )
    parser.add_argument('--output', type=Path, required=True, help='Output CSV path')
    args = parser.parse_args()

    gage_ids = load_gage_ids(args.events, args.staid_col)
    log.info('Found %d unique gages in %s', len(gage_ids), args.events)

    if args.raw_cache and args.raw_cache.exists():
        log.info('Loading cached raw download: %s', args.raw_cache)
        raw = pd.read_csv(args.raw_cache, dtype={'STAID': str})
    else:
        raw = download_all(gage_ids, args.start, args.end, args.parameter_code)
        if args.raw_cache:
            args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(args.raw_cache, index=False)
            log.info('Cached raw download: %s', args.raw_cache)

    discharge = to_utc_15min(raw, args.start, args.end)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    discharge.to_csv(args.output, index=False)
    log.info(
        'Wrote %s (%d rows, %d gages)',
        args.output,
        len(discharge),
        discharge['STAID'].nunique(),
    )


if __name__ == '__main__':
    main()
