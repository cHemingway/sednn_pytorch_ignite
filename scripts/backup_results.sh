#!/bin/bash

# Chris Hemingway 2019, MIT License
# Complex bash one liner, copies result_dir into previous/yy-mm-dd-hh-mm.tar.gz

if [ "$#" -ne 1 ]; then
    echo "Usage $0 result_dir"
    exit
fi

# For naming a tar file with todays date, see https://stackoverflow.com/a/18498409
name=$(date '+%F-%H-%M-%S')
# Create results dir if it does not exist
mkdir -p $1/previous
# Tar, using the name we created, _excluding_ the "previous" dir
shopt -s extglob # See https://unix.stackexchange.com/a/164026
tar -zcvf "$1/previous/$name.tar.gz" $1/!(previous)