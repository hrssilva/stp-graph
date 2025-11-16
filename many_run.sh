#!/usr/bin/env bash

n=100
m=20

# Initialize an empty array
arr=()

# Generate array elements
for (( i=1; i<=n; i++ )); do
    arr+=($((i * m)))
done

# Print the array
sizes="${arr[@]}"


python stp_benchmarks.py --use-numpy --max-fw-n 9999999\
    --repeats 5 \
    --density-dense 0.95 \
    --dense-sizes $sizes \
    --sparse-sizes $sizes \
    --csv "concurrent_many/stp_benchmarks.csv"

python stp_plots.py \
    --csv "concurrent_many/stp_benchmarks.csv" \

