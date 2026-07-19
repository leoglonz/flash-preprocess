r"""Patch corrected precipitation into mrms_15min.nc for the events flagged by
check_precip_gaps.py and re-extracted by rebuild_flagged_events_precip.py.

For each event in mrms_15min_reextracted.nc, overwrites that event's P values
in mrms_15min.nc (matched by storm_id, with catchments aligned by divide_id --
the two files' catchment axes are not identically ordered/sized). Every other
variable (n_steps, ts_start, ts_end, divide_id) and every non-flagged event's
P values are copied through unchanged.

Reads both files with raw netCDF4 (not xarray) to avoid any CF time
auto-decode/round-trip risk on ts_start/ts_end, which aren't being modified.

Writes to a new file first, then atomically replaces the original -- a
timestamped backup of the original is expected to already exist (see
mrms_15min.nc.bak_* alongside it) before this is run.

Run with:
    python engine/forcing/mrms/patch_mrms_15min.py
"""

import os
from pathlib import Path

import numpy as np
import netCDF4


# CONFIG
ORIG_NC = Path('/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min.nc')
PATCH_NC = Path(
    '/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min_reextracted.nc',
)


def main() -> None:
    """Patch corrected precipitation from PATCH_NC into ORIG_NC."""
    backups = sorted(ORIG_NC.parent.glob(f'{ORIG_NC.name}.bak_*'))
    if not backups:
        raise SystemExit(
            f'No backup found for {ORIG_NC} (expected {ORIG_NC}.bak_*) '
            '-- back it up before patching.',
        )
    print(f'Backup confirmed: {backups[-1]}')

    orig = netCDF4.Dataset(ORIG_NC, 'r')
    patch = netCDF4.Dataset(PATCH_NC, 'r')

    orig_storm_ids = np.array([str(int(s)) for s in orig.variables['storm_id'][:]])
    orig_divides = np.array(orig.variables['divide_id'][:], dtype=str)
    orig_idx_by_storm = {s: i for i, s in enumerate(orig_storm_ids)}
    orig_idx_by_divide = {d: j for j, d in enumerate(orig_divides)}

    patch_storm_ids = np.array(patch.variables['storm_id'][:], dtype=str)
    patch_divides = np.array(patch.variables['divide_id'][:], dtype=str)
    missing_divides = [d for d in patch_divides if d not in orig_idx_by_divide]
    if missing_divides:
        raise SystemExit(
            f'{len(missing_divides)} patch catchment(s) not found in '
            f'original divide_id axis: {missing_divides[:5]}',
        )
    patch_cols = np.array([orig_idx_by_divide[d] for d in patch_divides])

    P_orig = orig.variables['P'][:]  # masked array, (event, time_step, catchment)
    P_orig = np.ma.filled(P_orig, np.nan).astype('float32')
    P_patch = np.ma.filled(patch.variables['P'][:], np.nan).astype('float32')

    n_patched, n_missing = 0, 0
    for k, sid in enumerate(patch_storm_ids):
        i = orig_idx_by_storm.get(sid)
        if i is None:
            n_missing += 1
            continue
        P_orig[i, :, patch_cols] = P_patch[k, :, :].T
        n_patched += 1

    print(
        f'Patched {n_patched} events ({n_missing} storm_id from patch file not '
        f'found in original)',
    )

    tmp_path = str(ORIG_NC) + '.tmp'
    nc_out = netCDF4.Dataset(tmp_path, 'w', format='NETCDF4')
    n_event = orig.dimensions['event'].size
    n_time = orig.dimensions['time_step'].size
    n_cat = orig.dimensions['catchment'].size
    nc_out.createDimension('event', n_event)
    nc_out.createDimension('time_step', n_time)
    nc_out.createDimension('catchment', n_cat)

    v = nc_out.createVariable('storm_id', 'i4', ('event',))
    v.long_name = orig.variables['storm_id'].long_name
    v[:] = orig.variables['storm_id'][:]

    v = nc_out.createVariable('n_steps', 'i4', ('event',))
    v.long_name = orig.variables['n_steps'].long_name
    v[:] = orig.variables['n_steps'][:]

    v = nc_out.createVariable('ts_start', 'f8', ('event',))
    v.units = orig.variables['ts_start'].units
    v.long_name = orig.variables['ts_start'].long_name
    v[:] = orig.variables['ts_start'][:]

    v = nc_out.createVariable('ts_end', 'f8', ('event',))
    v.units = orig.variables['ts_end'].units
    v.long_name = orig.variables['ts_end'].long_name
    v[:] = orig.variables['ts_end'][:]

    v = nc_out.createVariable('divide_id', str, ('catchment',))
    v.long_name = orig.variables['divide_id'].long_name
    v[:] = orig_divides.astype(object)

    v_data = nc_out.createVariable(
        'P',
        'f4',
        ('event', 'time_step', 'catchment'),
        fill_value=np.nan,
        zlib=True,
        complevel=4,
        chunksizes=(min(16, n_event), n_time, min(64, n_cat)),
    )
    v_data.units = orig.variables['P'].units
    v_data.long_name = (
        'MRMS precipitation depth (patched: re-extracted for flagged events)'
    )
    v_data.coordinates = orig.variables['P'].coordinates
    v_data[:] = P_orig

    nc_out.close()
    orig.close()
    patch.close()

    os.replace(tmp_path, ORIG_NC)
    print(f'Done -> {ORIG_NC}')


if __name__ == '__main__':
    main()
