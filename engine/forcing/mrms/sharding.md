Using top 15 huc8s as example.

Step 1 — split events into shards (run once):


python /projects/mhpi/leoglonz/sub_hourly/flash_preprocess/engine/forcing/mrms/split_events_shards.py \
    --events-csv /projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/events.csv \
    --n-shards 8 \
    --out-dir /projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/shards
Step 2 — one command per terminal (8 shards, 50 workers each = 400 total):


# terminal 1
python /projects/mhpi/leoglonz/sub_hourly/flash_preprocess/engine/forcing/mrms/run_pipeline.py \
    --events-csv /projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/shards/events_shard0.csv \
    --tag-suffix _huc8top15_shard0 \
    --cache-dir /projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/neuse \
    --max-workers 50

# terminal 2
python /projects/mhpi/leoglonz/sub_hourly/flash_preprocess/engine/forcing/mrms/run_pipeline.py \
    --events-csv /projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/shards/events_shard1.csv \
    --tag-suffix _huc8top15_shard1 \
    --cache-dir /projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/neuse \
    --max-workers 50
...same pattern for shard2 through shard7 in terminals 3–8 (just increment the shard number in both --events-csv and --tag-suffix).

Step 3 — once all 8 finish, merge into the final file:


python /projects/mhpi/leoglonz/sub_hourly/flash_preprocess/engine/forcing/mrms/merge.py \
    /projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/neuse/vpu_runs/*_huc8top15_shard*/mrms_15min_part.nc \
    --out /projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/mrms_15min.nc
--cache-dir reuses the existing hydrofabric/crosswalk caches already built (no priming needed), and --tag-suffix keeps each shard's per-VPU files from colliding since VPU 02 shows up in every shard. Adjust 8/50 if you want a different split.