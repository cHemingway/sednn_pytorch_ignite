#!/bin/bash
# Script to resample all audio to 16KHz in advance to save time
# Chris Hemingway 2019, MIT License
#
# Requires GNU parallel, please ensure you cite this in your research as they request
set -e #Exit on first error
cd $1
find ./ -type f -name '*.wav' | parallel --will-cite --progress ffmpeg -hide_banner -y -loglevel error -i {} -af aresample=resampler=soxr -ar 16000 {}
echo "Done!"
