#!/bin/bash
set -xeuo pipefail

function do_plot {
    csv_file="$1"
    png_file="${csv_file/.csv/.png}"
    shift
    ./plot.py --output "$png_file" "$@" "$csv_file"
}

for csv_file in data-out/ex1/*.csv; do
    do_plot "$csv_file" --plot-traj-2d
done

for csv_file in data-out/ex2/*.csv; do
    do_plot "$csv_file" --plot-traj-2d --controls
done

for csv_file in data-out/ex3/*.csv; do
    do_plot "$csv_file" --plot-traj-3d
done

for a_csv_file in data-out/ex4/a-*.csv; do
    base_csv_file="${a_csv_file/a-with/with}"
    b_csv_file="${base_csv_file/with/b-with}"
    png_file="${base_csv_file}.png"
    ./plot.py --output "$png_file" --plot-norm --controls "$a_csv_file" "$b_csv_file"
done
