#!/usr/bin/env python
'''
Small script to slim down all noisy data to only one per file.
Uses symlinks rather than copy to speed up and save disk space.
'''
# Chris Hemingway 2019

import pathlib
import os, sys
import argparse
import random
import shutil

from tqdm import tqdm


def link(output_name, original):
    ''' Symlink wrapper that ensures absolute path and deletes old symlink '''
    # Make absolute, as pathlib.Path().absolute is deprecated
    # See https://bugs.python.org/issue29688
    output_name = os.path.abspath(output_name)
    output_name = pathlib.Path(output_name)
    # Delete old output name
    if output_name.is_symlink():
        output_name.unlink()
    # Finally, symlink
    output_name.symlink_to(original.resolve()) # Symlink to absolute path


def recreate_dir(folder):
    ''' Create a dir, deleting it if it already exists and recreating '''
    # We use recreate_dir instead of mkdir(exists_ok=True) as then we can
    # ensure that any old data is deleted.
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()


def create_set(clean_names, noisy_in_dir, clean_out_dir, noisy_out_dir):
    ''' Create a training set by randomly choosing noisy versions of clean_names '''
    # Recreate output dirs
    recreate_dir(clean_out_dir); recreate_dir(noisy_out_dir)
    # Loop through each clean name, picking names
    for name in tqdm(clean_names, desc="Picking noisy mixtures for SEGAN"):
        basename = name.stem # Get name without extension
        noisy_names = noisy_in_dir.glob(f"{basename}*?.wav")
        noisy_names = list(noisy_names) # Convert from generator
        if len(noisy_names) == 0:
            raise FileNotFoundError(f"Could not find matching noisy version/s of {name}")
        # Choose a random noise
        noisy_name = random.choice(noisy_names)
        # Generate name of symlink and point it to
        output_name = noisy_out_dir / (basename+"_noisy.wav")
        link(output_name, noisy_name)
        # Link clean name
        link(clean_out_dir/name.name, name)


def main(args):
    ''' Main function '''
    random.seed(args.random_seed)
    
    # Try opening noisy_dir, clean_dir
    clean_dir = pathlib.Path(args.clean_dir)
    noisy_dir = pathlib.Path(args.noisy_dir)

    # Try creating output folder
    train_dir = pathlib.Path(args.train_dir)
    recreate_dir(train_dir)
    validation_dir = pathlib.Path(args.validation_dir)
    recreate_dir(validation_dir)

    # Get clean audio names
    clean_names = clean_dir.glob("*.wav")
    clean_names = list(clean_names) # Convert from generator for TQDM

    # Split into test set and validation set names
    # Important to randomize before split to avoid alphabetic order of names bias
    random.shuffle(clean_names)
    split = args.validation_size
    if len(clean_names) < split:
        # Hack: Work with mini_data by using minimum of 25% or "1" as validation
        split = max(1, int(len(clean_names)) / 4)
    validation_names = clean_names[:split]
    train_names      = clean_names[split:]

    print(validation_names)
    
    # Create train set
    create_set(train_names, noisy_dir, train_dir/'clean', train_dir/'noisy')
    # Create validation set
    create_set(validation_names, noisy_dir, validation_dir/'clean', validation_dir/'noisy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                         help="Clean audio directory, won't be modified")
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help="Noisy audio directory")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Output directory for train symlinks, will remove all older contents!")
    parser.add_argument("--validation_dir", type=str, required=True,
                        help="Output directory for validation symlinks, will remove all older contents!")
    parser.add_argument("--random_seed", default=None, required=False,
                        help="Seed for randomly selecting which noise to keep")
    parser.add_argument("--validation_size", default=20,
                        help="Size of validation set")

    args = parser.parse_args()
    main(args)