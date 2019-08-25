"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: - 
"""
import os
import numpy as np
import argparse
import csv
import time
import logging
from typing import Tuple
import functools

from tqdm import tqdm, trange
import h5py
import pickle

from utils.utilities import (create_folder, read_audio, write_audio, 
    calculate_spectrogram, log_sp, mat_2d_to_3d, pad_with_border, 
    calculate_scaler, save_features, load_features)
import utils.config as config

from MRCG_python import MRCG as MRCG

import multiprocessing

# ! HACK: fix onset of noise to zero
FIXED_NOISE_ONSET = True

# Disable multiprocessing here for easy debug
USE_MULTIPROCESSING = True

## Utility functions
def get_audio_length(audio_path):
    (audio, _) = read_audio(audio_path)
    return len(audio)


def get_rand_onset_offset(len_speech, len_noise, rs=np.random.RandomState()):
    ''' Given a length of noise and a length of speech, 
        select a random onset/offset of the noise equal to the speech length '''

    if FIXED_NOISE_ONSET:
        onset = 0
        offset = len_speech

    else:
        # If noise shorter than speech then noise will be repeated in calculating features
        if len_noise <= len_speech:
            onset = 0
            offset = len_speech
        # If noise longer than speech then randomly select a segment of noise
        else:
            onset = rs.randint(0, len_noise - len_speech, size=1)[0]
            offset = onset + len_speech

    return onset, offset

###
def create_mixture_csv(args):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    magnification = args.magnification
    extra_speakers = args.extra_speakers
    
    rs = np.random.RandomState(0) # Use the same seed every time!
    
    # Paths
    out_csv_path = os.path.join(workspace, 'mixture_csvs', '{}.csv'.format(data_type))
    create_folder(os.path.dirname(out_csv_path))
    
    speech_names = [name for name in os.listdir(speech_dir) if name.lower().endswith('.wav')]
    noise_names = [name for name in os.listdir(noise_dir) if name.lower().endswith('.wav')]
    
    # Names for CSV File
    fieldnames = ['speech_name', 'noise_name', 'noise_onset', 'noise_offset']
    extra_speaker_fields = ['extra_speech_name{}', 'extra_speech_onset{}', 'extra_speech_offset{}']
    if extra_speakers:
        for n in range(extra_speakers):
            fieldnames += [field.format(n) for field in extra_speaker_fields]

    with open(out_csv_path, 'w') as f:
        # Open CSV file writer. TODO use DictWriter
        writer = csv.writer(f, dialect='excel')
        writer.writerow(fieldnames) # Write header

        for speech_na in tqdm(speech_names,desc="Creating Mixture CSV"):
            
            # Read speech
            speech_path = os.path.join(speech_dir, speech_na)
            (speech_audio, _) = read_audio(speech_path)
            len_speech = len(speech_audio)
            
            # For training data, mix each speech with randomly picked #magnification noises
            if data_type == 'train':
                selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
                
            # For test data, mix each speech with all noises
            elif data_type == 'test':
                selected_noise_names = noise_names
            else:
                raise Exception('data_type must be train | test!')

            # Pick extra speech names, ensuring we don't select our own
            extra_speech_names = []
            while len(extra_speech_names) < extra_speakers:
                name = rs.choice(speech_names)
                if name != speech_na:
                    extra_speech_names.append(name)

            # Mix one speech with different noises many times
            for noise_na in selected_noise_names:

                # Choose onset and offset of noise
                noise_len = get_audio_length(os.path.join(noise_dir, noise_na))
                noise_onset, noise_offset = get_rand_onset_offset(len_speech, noise_len, rs)

                # Choose onset and offset of each random speaker
                extra_fields = []
                for name in extra_speech_names:
                    len_extra = get_audio_length(os.path.join(speech_dir, name))
                    onset, offset = get_rand_onset_offset(len_speech, len_extra, rs)
                    # Generate extra fields
                    extra_fields += [name, onset, offset]

                # Write out row to CSV
                writer.writerow([speech_na, noise_na, noise_onset, noise_offset] +
                                extra_fields)
    
    # Finished, log name of CSV file
    logging.info('Write {} mixture csv to {}!'.format(data_type, out_csv_path))


def calculate_features(args, row):
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    snr = args.snr
    extra_speech_db = args.extra_speech_db # SNR of extra speakers
    sample_rate = config.sample_rate

    noise_onset = int(row['noise_onset'])
    noise_offset = int(row['noise_offset'])

    def read_speech(name):
        speech_path = os.path.join(speech_dir, name)
        (speech_audio, _) = read_audio(speech_path, target_fs=sample_rate)
        return speech_audio
    
    # Read speech audio
    speech_audio = read_speech(row['speech_name'])
    
    # Read noise audio
    noise_path = os.path.join(noise_dir, row['noise_name'])
    (noise_audio, _) = read_audio(noise_path, target_fs=sample_rate)
    
    # Trim/repeat noise
    noise_audio = adjust_noise_length(noise_audio, noise_onset, noise_offset, speech_audio)

    # Scale speech to given snr
    scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
    speech_audio *= scaler

    # Read extra speech audio
    extra_speech_total = np.zeros_like(speech_audio)
    for name,onset,offset in get_extras(row):
          extra_speech = read_speech(name)
          extra_speech = adjust_noise_length(extra_speech, onset, offset, speech_audio)
          scaler = get_amplitude_scaling_factor(extra_speech, speech_audio, snr=extra_speech_db)
          extra_speech *= scaler # Scale the extra speech instead
          extra_speech_total += extra_speech
    
    # Add to noise_audio
    noise_audio += extra_speech_total
        
    # Get normalized mixture, speech, noise
    (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)

    # Write out mixed audio
    out_bare_name = os.path.join('{}.{}'.format(
        os.path.splitext(row['speech_name'])[0], os.path.splitext(row['noise_name'])[0]))
        
    out_audio_path = os.path.join(workspace, 'mixed_audios', 
        data_type, '{}db'.format(int(snr)), '{}.wav'.format(out_bare_name))
        
    create_folder(os.path.dirname(out_audio_path))
    write_audio(out_audio_path, mixed_audio, sample_rate)
    if logging.DEBUG >= logging.root.level:
        tqdm.write('Write mixture wav to: {}'.format(out_audio_path))

    # Extract spectrogram
    mixed_complx_x = calculate_spectrogram(mixed_audio, mode='complex')
    speech_x = calculate_spectrogram(speech_audio, mode='magnitude')
    noise_x = calculate_spectrogram(noise_audio, mode='magnitude')
    
    # Extract MRCG on request
    if args.mrcg:
        # Make sure MRCG is the same length as the sample rate
        mrcg_window = config.window_size
        cochs, del0, ddel = MRCG.mrcg_extract_components(mixed_audio, 
                                                         sample_rate, 
                                                         window_len = mrcg_window)
        mrcg = cochs # Use only cochleagrams, not differences
    else:
        mrcg = None

    # Write out features
    out_feature_path = os.path.join(workspace, 'features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), '{}.npz'.format(out_bare_name))
        
    create_folder(os.path.dirname(out_feature_path))
    save_features(out_feature_path,mixed_complx_x, speech_x, noise_x, alpha, mrcg)
    

###
def calculate_mixture_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    # Further args get passed to calculate_features function

    if args.mrcg:
        logging.warning("Calculating MRCG as well")
    
    # Paths
    mixture_csv_path = os.path.join(workspace, 'mixture_csvs', '{}.csv'.format(data_type))
    
    # Open mixture csv and convert to list of rows
    with open(mixture_csv_path, 'r') as f:
        reader = csv.DictReader(f, dialect='excel')
        rows = list(reader)

    # Create progress bar
    pbar = tqdm(desc="Calculating {} features".format(data_type), total=len(rows))

    # Create partial function of calculate_features so only rows var is changed
    apply_row = functools.partial(calculate_features, 
                                  args
                                  #rows is passed here
                                  )

    if USE_MULTIPROCESSING:
        # Use multiprocessing to call calculate_features on each row of CSV
        # We use imap to get a progress bar here, see https://github.com/tqdm/tqdm/issues/484#issuecomment-351001534
        pool = multiprocessing.Pool()

        for _ in pool.imap_unordered(apply_row, rows):
            pbar.update() # Update progress bar

        pool.close()
        pool.join() # Block until all finished
        pbar.close() # Close progress bar

    else:
        # Single threaded for debugging
        for r in rows:
            apply_row(r)
            pbar.update()

    

def get_extras(row: dict) -> Tuple[str, int, int]:
    """Generator that returns the extra speech details
    Args:
        row (dict): The csv.DictReader row
    Returns:
        Tuple[str, int, int]: Name, onset, offset
    """
    for k in row.keys():
        if "extra_speech_name" in k:
            n = int(k.lstrip("extra_speech_name")) # Number is straight after
            name = row[k]
            onset = int(row[f"extra_speech_onset{n}"])
            offset = int(row[f"extra_speech_offset{n}"])
            yield (name, onset, offset)


def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor


def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n
        
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha


def adjust_noise_length(noise_audio, onset, offset, speech_audio):
    ''' Pads or truncates noise audio to match length of speech audio '''
    if noise_audio.size < speech_audio.size:
        # Repeat noise to the same length as speech
        n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
        noise_audio_repeat = np.tile(noise_audio, n_repeat)
        noise_audio = noise_audio_repeat[0 : len(speech_audio)]
    else:
        # Truncate noise to the same length as speech
        noise_audio = noise_audio[onset: offset]
    return noise_audio
    
    
###
def pack_features(args):
    """Load all features, apply log and conver to 3D tensor: 
    (samples, n_concat, freq_bins) and write it out to an hdf5 file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop
    
    # Paths
    feature_dir = os.path.join(workspace, 'features', 'spectrogram', data_type, 
        '{}db'.format(int(snr)))
        
    hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), 'data.h5')
        
    create_folder(os.path.dirname(hdf5_path))
    
    x_all = []  # (n_segs, n_concat, n_freq)
    mrcg_all = [] # (n_segs, coch1-4, ch0-64)
    y_all = []  # (n_segs, n_freq)
    

    # Load all features
    names = os.listdir(feature_dir)
    
    for name in tqdm(names, desc="Packing features:"):
        
        # Load feature. 
        feature_path = os.path.join(feature_dir, name)
        data = load_features(feature_path)
        mixed_complx_x, speech_x, noise_x, alpha, mrcg = data

        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values. 
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
    
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        mixed_x_3d = log_sp(mixed_x_3d).astype(np.float32)
        x_all.append(mixed_x_3d)
        
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, (n_concat - 1) // 2, :]
        y = log_sp(y).astype(np.float32) 
        y_all.append(y)

        # MRCG is of shape (CGn, (ch, time))
        # Need to stack and reorder to (time, CGn, ch)
        if mrcg is not None:
            mrcg_3d = []
            for cg in mrcg:
                cg = cg.T# Transpose (ch, time) to (time, ch)
                cg = pad_with_border(cg, n_pad) # Pad each cochleagram
                # Concat and unconcat to get same padding logic as x, y
                mrcg_cg_3d = mat_2d_to_3d(cg, agg_num=n_concat, hop=n_hop)
                mrcg_cg_2d = mrcg_cg_3d[:, (n_concat - 1) // 2, :]
                mrcg_3d.append(mrcg_cg_2d)
            mrcg_3d = np.stack(mrcg_3d, axis=0)
            mrcg_3d = np.swapaxes(mrcg_3d, 0 , 1) 
            mrcg_3d = mrcg_3d.astype(np.float32) # Half the disk
            mrcg_all.append(mrcg_3d)

    x_all = np.concatenate(x_all, axis=0)   # (n_segs, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
    mrcg_all = np.concatenate(mrcg_all, axis=0) # (n_segs, cg_n, n_ch)

    # Write out data to hdf5 file. 
    logging.debug("Saving..")
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        hf.create_dataset('mrcg', data=mrcg_all)
    
    logging.info('Write out to {}'.format(hdf5_path))
    

###
def write_out_scaler(args):
    """Compute and write out scaler of data. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    
    # Paths
    hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), 'data.h5')
    
    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), 'scaler.p')
        
    create_folder(os.path.dirname(scaler_path))
    
    # Load data
    t1 = time.time()
    
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['x'][:]     # (n_segs, n_concat, n_freq)
    
    # Compute scaler
    scaler = calculate_scaler(x, axis=(0, 1))
    
    if args.print_scalar or (logging.root.level == logging.DEBUG):
        print(scaler)
    
    # Write out scaler
    pickle.dump(scaler, open(scaler_path, 'wb'))
    
    logging.debug('Save scaler to {}'.format(scaler_path))
    logging.info('Compute scaler finished! {} s'.format(time.time() - t1,))
    
    
###
if __name__ == '__main__':

    # Needed for PTVSD Debugging of multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--debug',
        help="Show all debug information",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Show more messages",
        action="store_const", dest="loglevel", const=logging.INFO,
    )

    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)
    parser_create_mixture_csv.add_argument('--extra_speakers', type=int, default=0)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    parser_calculate_mixture_features.add_argument('--extra_speech_db', type=float, default=-5)
    parser_calculate_mixture_features.add_argument('--mrcg', action='store_true', default=False)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    parser_write_out_scaler = subparsers.add_parser('write_out_scaler')
    parser_write_out_scaler.add_argument('--workspace', type=str, required=True)
    parser_write_out_scaler.add_argument('--data_type', type=str, required=True)
    parser_write_out_scaler.add_argument('--snr', type=float, required=True)
    parser_write_out_scaler.add_argument('--print_scalar', action='store_true', default=False)
    
    args = parser.parse_args()

    # Set logging verbosity
    logging.basicConfig(level=args.loglevel)
    
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
        
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
        
    elif args.mode == 'pack_features':
        pack_features(args)       
        
    elif args.mode == 'write_out_scaler':
        write_out_scaler(args)
        
    else:
        raise Exception('Error!')