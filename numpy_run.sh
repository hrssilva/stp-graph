#!/usr/bin/env bash

n=12
m=50

# Initialize an empty array
arr=()

# Generate array elements
for (( i=0; i<n; i++ )); do
    arr+=($(((2**i) + (m*i))))
done

# Print the array
sizes="${arr[@]}"


python stp_benchmarks.py --use-numpy --max-fw-n 9999999\
    --repeats 5 \
    --density-dense 0.95 \
    --dense-sizes $sizes \
    --sparse-sizes $sizes \
    --csv "concurrent_numpy/stp_benchmarks.csv"
    #--dense-sizes 40 60 80 100 140 180 220 300 380 460 560 660 760 900 1040 1180 \
    #--sparse-sizes 40 60 80 100 140 180 220 300 380 460 560 660 760 900 1040 1180
