#!/usr/bin/env bash

python stp_benchmarks.py --no-show --max-fw-n 1200\
    --repeats 5 \
    --density-dense 0.95 \
    --dense-sizes 40 60 80 100 140 180 220 300 380 460 560 660 760 900 1040 1180 \
    --sparse-sizes 40 60 80 100 140 180 220 300 380 460 560 660 760 900 1040 1180
