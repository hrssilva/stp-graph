#!/usr/bin/env bash

python stp_benchmarks.py --no-show --max-fw-n 800\
    --dense-sizes 40 60 80 100 140 180 220 300 380 460 560 660 760 \
    --sparse-sizes 40 60 80 100 140 180 220 300 380 460 560 660 760 
