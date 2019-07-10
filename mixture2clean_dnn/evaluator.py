'''
Compares folders of clean and dirty audio
Calculates STOI, PESQ (tbd) and MSS
'''
import pathlib
import typing
from typing import Union
from dataclasses import dataclass

import numpy as np  # type: ignore

import soundfile  # type: ignore
from mir_eval.separation import bss_eval_sources  # type: ignore
from pystoi.stoi import stoi


@dataclass
class Metrics:
    """ dataclass for metrics for each file
    """
    sdr: np.float_
    sir: np.float_
    sar: np.float_
    stoi: np.float_


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

    # Calculate STOI, original version
    d = stoi(clean, dirty, dirty_fs, extended=False)

    # Calculate BSS statistics
    # Use compute_permutation as this is what bss did
    [sdr, sir, sar, _] = bss_eval_sources(
        clean, dirty, compute_permutation=True)

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
