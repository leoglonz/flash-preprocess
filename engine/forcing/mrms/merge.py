import argparse

from flash_preprocess.mrms import merge_parts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge per-VPU mrms_15min_part.nc files into one NetCDF.")
    ap.add_argument("parts", nargs="+", help="per-VPU mrms_15min_part.nc files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge_parts(args.parts, args.out)
