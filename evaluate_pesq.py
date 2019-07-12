"""
Summary:  Calculate PESQ and overal stats of enhanced speech. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: 2019.07.12 - Removed get_stats
"""
import sys
import argparse
import os
import subprocess
import csv
import numpy as np

from tqdm import tqdm


def calculate_pesq(args):
    """Calculate PESQ of all enhaced speech. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of clean speech. 
      te_snr: float, testing SNR. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    enh_speech_dir = args.enh_speech_dir
    te_snr = args.te_snr
    
    # Remove already existed file. 
    try:
        os.remove('_pesq_itu_results.txt')
        os.remove('_pesq_results.txt')
    except FileNotFoundError:
        pass # File does not exist, so no need to delete it
    
    # Calculate PESQ of all enhaced speech. 
    if not enh_speech_dir:
        enh_speech_dir = os.path.join(workspace, 'enh_wavs', 'test', '{}db'.format(int(te_snr)))
        
    names = os.listdir(enh_speech_dir)
    for (cnt, na) in enumerate(tqdm(names,desc="Calculating PESQ")):
        if args.show_names: # Show name of original file on request
            tqdm.write(na)
        enh_path = os.path.join(enh_speech_dir, na)
        
        # TODO: Work with both upper and lower case .wav and .WAV
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, '{}.wav'.format(speech_na))
        
        # Call executable PESQ tool, hiding output
        cmd = ' '.join(['./pesq', speech_path, enh_path, '+16000'])
        subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--enh_speech_dir', type=str, default=None)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    parser_calculate_pesq.add_argument('--show_names',action='store_true',default=False)
    args = parser.parse_args()
        
    if args.mode == 'calculate_pesq':
        calculate_pesq(args)
        
    elif args.mode == 'get_stats':
        sys.exit('get_stats has been removed')
        
    else:
        raise Exception('Incorrect argument!')