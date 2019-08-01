''' Tasks for data preperation, mixing audio, extracting features etc '''

import os
import pathlib

from doit import create_after
from doit.tools import (Interactive, PythonInteractiveAction, config_changed,
                        title_with_actions)

# HACK: Ideally would fix import paths instead
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

#pylint: disable=wrong-import-position
from utils import prepare_data
from tasks.utils import get_source_files, get_data_filenames

# Needed to get around the args nonsense
class DictAttr(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

#
# Wrapper functions to pack args
#
def mix_csv(workspace, subdata, data_type, magnification=1):
    args = {
        "workspace": workspace,
        "speech_dir": subdata["speech"],
        "noise_dir": subdata["noise"],
        "data_type": data_type,
        "magnification": magnification
    }
    prepare_data.create_mixture_csv(DictAttr(args))


def mix_features(workspace, subdata, data_type, snr):
    args = {
        "workspace": workspace,
        "speech_dir": subdata["speech"],
        "noise_dir": subdata["noise"],
        "data_type": data_type,
        "snr": snr
    }
    prepare_data.calculate_mixture_features(DictAttr(args))


def pack_features(workspace, data_type, snr, n_concat, n_hop):
    args = {
        "workspace": workspace,
        "data_type": data_type,
        "snr": snr,
        "n_concat": n_concat,
        "n_hop": n_hop
    }
    prepare_data.pack_features(DictAttr(args))


def write_out_scalar(workspace, data_type, snr):
    args = {
        "workspace": workspace,
        "data_type": data_type,
        "snr": snr,
        "print_scalar": False
    }
    prepare_data.write_out_scaler(DictAttr(args))
   


class Data_Prepare_creator(object):

    def __init__(self, data, workspace_dir,
                 scalar_path, packed_feature_paths, config):
        self.data = data
        self.workspace = workspace_dir
        self.scalar_path = scalar_path
        self.packed_feature_paths = packed_feature_paths
        self.config = config

        # Find names of data files
        self.data_files = get_data_filenames(data)

    def create_mixture_csv(self):
        # TODO replace with yielding multiple tasks
        return {
            'name': 'create_mixture_csv',
            'file_dep':  self.data_files + get_source_files("utils"),
            'task_dep': ['make_workspace'],
            # Using pathlib slash '/' operator
            'targets': [
                self.workspace / 'mixture_csvs' / 'test.csv',
                self.workspace / 'mixture_csvs' / 'train.csv'
            ],
            # Call mix_csv on each type of data
            # Need to wrap in PythonInteractiveAction for TQDM to work
            'actions': [
                PythonInteractiveAction(
                    mix_csv, [self.workspace, self.data["train"], "train", self.config["magnification"]]),
                PythonInteractiveAction(
                    mix_csv, [self.workspace, self.data["test"], "test", self.config["magnification"]]),
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    def calculate_mixture_features(self):
        # TODO replace with yielding multiple tasks
        return {
            'name': 'calculate_mixture_features',
            'file_dep':  self.data_files + get_source_files("utils") + self.create_mixture_csv()['targets'],
            'targets': [
                self.data['mixed'],
                self.workspace / "features",
            ],
            'actions': [
                PythonInteractiveAction(
                    mix_features, [self.workspace, self.data["train"], "train", self.config["train_snr"]]),
                PythonInteractiveAction(
                    mix_features, [self.workspace, self.data["test"], "test", self.config["test_snr"]]),
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    @create_after(executed='prepare_data:calculate_mixture_features', target_regex='.*\data.h5')
    def pack_features(self):
        shared_args = [self.config["n_concat"], self.config["n_hop"]]

        feature_path = self.workspace / 'features'
        features = list(feature_path.rglob("*.p"))  # Search for all .p files

        return {
            'name': 'pack_features',
            'file_dep': features + get_source_files("utils"), 
            'task_dep': ['prepare_data:calculate_mixture_features'], # TODO use files instead
            'targets': self.packed_feature_paths,
            'actions': [
                PythonInteractiveAction(
                    pack_features, [self.workspace, "train", self.config["train_snr"], *shared_args]),
                PythonInteractiveAction(
                    pack_features, [self.workspace, "test", self.config["test_snr"], *shared_args]),
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    def write_out_scalar(self):
        return {
            'name': 'write_out_data',
            'file_dep':  self.packed_feature_paths + get_source_files("utils"),
            'targets': [
                self.scalar_path
            ],
            'actions': [
                PythonInteractiveAction(
                    write_out_scalar, [self.workspace, "train", self.config["train_snr"]]),
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    def tasks(self):
        ''' Mix audio and extract features '''
        for task in [self.create_mixture_csv, self.calculate_mixture_features,
                     self.pack_features, self.write_out_scalar]:
            yield task()
                     
