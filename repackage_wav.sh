#!/bin/bash
# Script to repackage all wav files, keeping audio the same
# Chris Hemingway 2019, MIT License
#
INDIR="metadata"
OUTDIR="metadata_repackaged" # Presumes on same level as INDIR
set -e #Exit on first error
echo "Creating directory $OUTDIR"
mkdir -p flac48 # Make empty folder
cd $INDIR # Move to wave folder
echo "Copying directory structure"
find . -type d -exec mkdir -p ../$OUTDIR/{} \; # Copy directory structure
echo "Converting to FLAC"
# Finally, convert the lot!
find ./ -type f -name '*.wav' | parallel --will-cite --progress ffmpeg -hide_banner -loglevel error -i {} -c:a copy ../$OUTDIR/{.}.wav
echo "Done!"
