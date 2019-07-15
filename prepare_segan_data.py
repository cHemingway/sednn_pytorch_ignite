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

from tqdm import tqdm


def main(args):
    ''' Main function '''
    random.seed(args.random_seed)
    
    # Try opening noisy_dir, clean_dir
    clean_dir = pathlib.Path(args.clean_dir)
    noisy_dir = pathlib.Path(args.noisy_dir)

    # Try creating output folder
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True) # Don't throw exception if exists

    # Get clean audio names
    clean_names = clean_dir.glob("*.wav")
    clean_names = list(clean_names) # Convert from generator for TQDM

    # For each clean name, choose a random noise
    for name in tqdm(clean_names, desc="Picking noisy mixtures for SEGAN"):
        basename = name.stem # Get name without extension
        dirty_names = noisy_dir.glob(f"{basename}*?.wav")
        dirty_names = list(dirty_names) # Convert from generator
        if len(dirty_names) == 0:
            raise FileNotFoundError(f"Could not find matching noisy version/s of {name}")
        # Choose a random noise
        dirty_name = random.choice(dirty_names)
        # Generate name of symlink and point it to
        output_name = output_dir / (basename+"_noisy.wav")
        output_name = output_name.resolve() # Make absolute
        output_name.symlink_to(dirty_name.resolve()) # Symlink to absolute path 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                         help="Clean audio directory, won't be modified")
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help="Noisy audio directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for symlinks, will remove all older contents!")
    parser.add_argument("--random_seed", default=None, required=False,
                        help="Seed for randomly selecting which noise to keep")

    args = parser.parse_args()
    main(args)