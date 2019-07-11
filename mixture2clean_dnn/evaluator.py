'''
Compares folders of clean and dirty audio
Calculates STOI, PESQ (tbd) and MSS
'''
import argparse
import pathlib
import typing
from typing import Union
from dataclasses import dataclass
import logging

from tqdm import tqdm  # type: ignore
import soundfile  # type: ignore
from mir_eval.separation import bss_eval_sources  # type: ignore
from pystoi.stoi import stoi  # type: ignore


@dataclass
class Metrics:
    """ dataclass for metrics for each file
    """
    sdr: float
    sir: float
    sar: float
    stoi: float


def evaluate_metrics(dirty_file: Union[str, typing.BinaryIO],
                     clean_file: Union[str, typing.BinaryIO]) -> Metrics:
    """Evaluate metrics for a dirty/clean file pair

    Args:
        dirty_file (Union[str, typing.BinaryIO]): A dirty/noisy audio file
        clean_file (Union[str, typing.BinaryIO]): The original clean file

    Raises:
        ValueError: If sample rates do not match

    Returns:
        Metrics: A Metrics dataclass
    """
    dirty, dirty_fs = soundfile.read(dirty_file)
    clean, clean_fs = soundfile.read(clean_file)

    if dirty_fs != clean_fs:
        raise ValueError("Files have different sample rates!")

    if (dirty.ndim > 1) or (clean.ndim > 1):
        raise ValueError("Files are not mono!")

    # HACK: Reduce length
    if len(dirty) > len(clean):
        logging.warning("File %s is different length %d from clean file %d",
                dirty_file, len(dirty), len(clean))
        dirty = dirty[:len(clean)]
    elif len(clean) > len(dirty):
        logging.warning("File %s is different length %d from clean file %d",
                dirty_file, len(dirty), len(clean))
        clean = clean[:len(dirty)]

    # Calculate STOI, original version
    d = stoi(clean, dirty, dirty_fs, extended=False)

    # Calculate BSS statistics
    # Use compute_permutation as this is what bss did
    [sdr, sir, sar, _] = bss_eval_sources(
        clean, dirty, compute_permutation=True)

    # Flatten out from numpy array as mono so single element
    sdr = sdr.item()
    sir = sir.item()
    sar = sar.item()

    # Return value
    return Metrics(sdr=sdr, sir=sir, sar=sar, stoi=d)


def matching_clean_file(dirty_filename, dirty_folder: Union[str, pathlib.Path])\
        -> pathlib.Path:
    """ Find matching clean filepath given dirty filename/path

    Args:
        dirty_filename (Union[str, pathlib.Path]): Name of the dirty file

    Returns:
        pathlib.Path: Path to the clean 
    """
    raise NotImplementedError


def compare_folder(clean_folder, dirty_folder) -> typing.Dict[str, Metrics]:
    """ Calculates metrics for a folder of clean and dirty audio

    Args:
        clean_folder ([type]): A folder (path or string) of the original audio
        dirty_folder ([type]): A folder of the dirty/recovered audio to compare

    Returns:
        typing.Dict[str,Metrics]: A dictionary of clean filename : Metrics
    """
    clean_folder = pathlib.Path(clean_folder)
    dirty_folder = pathlib.Path(dirty_folder)

    dirty_files = clean_folder.glob("*.wav")

    metrics = dict()

    for dirty_file in tqdm(dirty_files):
        clean_file = matching_clean_file(dirty_file, dirty_folder)
        file_metrics = evaluate_metrics(str(dirty_file), str(clean_file))

        name = dirty_file.name()  # Name excluding folder etc
        metrics[name] = file_metrics

    return metrics


def write_metrics(metrics_dict: typing.Dict[str, Metrics],
                  metrics_file: Union[str, pathlib.Path]) -> None:
    """ Write a dictionary of metrics to a specified file path """
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_speech', type=str, required=True)
    parser.add_argument('--dirty_speech', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False, default="metrics.csv")
    parser.add_argument('--show_names', action='store_true', default=False)
    args = parser.parse_args()

    metrics = compare_folder(args.clean_folder, args.dirty_folder)
    print(metrics)

    write_metrics(metrics, args.output_file)
