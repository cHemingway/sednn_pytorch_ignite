'''
Calculates statistics from csv files and output of pesq
'''

import sys
import argparse
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def split_names(data: pd.DataFrame) -> pd.DataFrame:
    ''' Splits names column '''
    # Split name column, first part before the '.' is the clean name
    split_names = data['name'].str.split('.', expand=True)
    # Get clean and noisy parts from names, as category types
    data['clean'] = split_names.get(0).astype('category')
    data['noise'] = split_names.get(1).astype('category')
    return data


def merge_stoi_pesq(stoi_data: pd.DataFrame, pesq_data: pd.DataFrame)\
        -> pd.DataFrame:
    pesq_data.rename(columns={'DEGRADED':'name'}, inplace=True)
    stoi_data['name'] = stoi_data['name'].str.strip() # Remove whitespace
    return pd.merge(stoi_data, pesq_data, on='name')


def stoi_relabel_columns(stoi_data: pd.DataFrame) -> pd.DataFrame:
    ''' Relabel STOI columns to upper case, excluding name column '''
    return stoi_data.rename(columns= lambda n: str.upper(n) if n!='name' else n)
    

def pesq_relabel_columns(pesq_data: pd.DataFrame) -> pd.DataFrame:
    ''' Remove unused PESQ columns e.g. 'COND' and relabel others '''
    pesq_data.rename(columns={'PESQMOS':'PESQ'}, inplace=True)
    return pesq_data.drop(columns=['SUBJMOS','COND','SAMPLE_FREQ'])


def generate_boxplot(data: pd.DataFrame):
    f, ax = plt.subplots(nrows=2,ncols=1,figsize=(7, 8))
    plt.subplot(2,1,1)
    stoi_ax = sns.boxplot(x='noise',y='STOI',data=data)
    plt.subplot(2,1,2)
    pesq_ax = sns.boxplot(x='noise',y='PESQ',data=data)
    f.add_subplot(stoi_ax)
    f.add_subplot(pesq_ax)
    return f

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--pesq_file', type=str, required=True)
    parser.add_argument('--plot_file', type=str, required=False, default=None)
    # TODO Add output file?
    args = parser.parse_args()

    # Try opening files, show error if not found
    try:
        stoi_data = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        sys.exit(f"Could not find CSV file {args.csv_file}")
    try:
        pesq_data = pd.read_csv(args.pesq_file,sep='\t',skipinitialspace=True)
    except FileNotFoundError:
        sys.exit(f"Could not find PESQ file {args.csv_file}")

    # Fixup PESQ and STOI columns and remove unused
    pesq_data = pesq_relabel_columns(pesq_data)
    stoi_data = stoi_relabel_columns(stoi_data)

    # Merge files based on name
    data = merge_stoi_pesq(stoi_data, pesq_data)

    # Split name into clean and dirty section
    data = split_names(data)

    # Show mean values
    print(data.groupby('noise').mean())

    # Generate boxplot
    if args.plot_file:
        f = generate_boxplot(data)
        f.savefig(args.plot_file)
