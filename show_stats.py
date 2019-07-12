'''
Calculates statistics from csv files and output of pesq
'''

import sys
import argparse
from typing import Dict

import numpy as np
import pandas as pd


def split_names(data: pd.DataFrame) -> pd.DataFrame:
    ''' Splits names column '''
    # Split name column, first part before the '.' is the clean name
    split_names = data['name'].str.split('.', expand=True)
    # Get clean and noisy parts from names, as category types
    data['clean'] = split_names.get(0).astype('category')
    data['noise'] = split_names.get(1).astype('category')
    return data


def merge_csv_pesq(csv_data: pd.DataFrame, pesq_data: pd.DataFrame)\
        -> pd.DataFrame:
    pesq_data.rename(columns={'DEGRADED':'name'}, inplace=True)
    csv_data['name'] = csv_data['name'].str.strip() # Remove whitespace
    return pd.merge(csv_data, pesq_data, on='name')


def pesq_remove_unused(pesq_data: pd.DataFrame) -> pd.DataFrame:
    ''' Remove unused PESQ columns e.g. 'COND' '''
    return pesq_data.drop(columns=['SUBJMOS','COND','SAMPLE_FREQ'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--pesq_file', type=str, required=True)
    parser.add_argument('--plot', action='store_true')
    # TODO Add output file?
    args = parser.parse_args()

    # Try opening files, show error if not found
    try:
        csv_data = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        sys.exit(f"Could not find CSV file {args.csv_file}")
    try:
        pesq_data = pd.read_csv(args.pesq_file,sep='\t',skipinitialspace=True)
    except FileNotFoundError:
        sys.exit(f"Could not find PESQ file {args.pesq_file}")

    # Remove unused PESQ columns
    pesq_data = pesq_remove_unused(pesq_data)

    # Merge files based on name
    data = merge_csv_pesq(csv_data, pesq_data)

    # Split name into clean and dirty section
    data = split_names(data)

    # Show mean values
    print(data.groupby('noise').mean())
