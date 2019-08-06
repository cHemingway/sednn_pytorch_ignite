''' Tasks for data preperation, mixing audio, extracting features etc '''

import os
import pathlib

from doit import create_after, get_var
from doit.tools import (Interactive, PythonInteractiveAction, config_changed,
                        title_with_actions)

from tasks.utils import get_source_files, get_data_filenames

DEBUG_ATTACH = get_var("attach",default=False)


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

    def create_mixture_csv(self,data_type):
        ''' Yields a create_mixture_csv task of data_type '''
        yield {
            'name': f'create_mixture_csv:{data_type}',
            'file_dep':  self.data_files + get_source_files("utils"),
            'task_dep': ['make_workspace'],
            # Using pathlib slash '/' operator
            'targets': [
                self.workspace / 'mixture_csvs' / f'{data_type}.csv',
            ],
            # Call mix_csv on each type of data
            # Need to wrap in PythonInteractiveAction for TQDM to work
            'actions': [
                Interactive(
                    f"python prepare_data.py create_mixture_csv "
                    f"--workspace={self.workspace} "
                    f"--speech_dir={self.data[data_type]['speech']} "
                    f"--noise_dir={self.data[data_type]['noise']} " 
                    f"--data_type={data_type} "
                    f"--magnification={self.config['magnification']}"
                )
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    def calculate_mixture_features(self, data_type):
        ''' Yields a calculate_mixture_features of data_type '''
        yield {
            'name': f'calculate_mixture_features:{data_type}',
            'task_dep': [f'prepare_data:create_mixture_csv:{data_type}'],
            'targets': [
                self.data['mixed']/data_type,
                self.workspace / "features"/'spectogram'/data_type,
            ],
            'actions': [
                Interactive(
                    f"python prepare_data.py calculate_mixture_features "
                    f"--workspace={self.workspace} "
                    f"--speech_dir={self.data[data_type]['speech']} "
                    f"--noise_dir={self.data[data_type]['noise']} " 
                    f"--data_type={data_type} "
                    f"--snr={self.config[f'{data_type}_snr']}"
                )
            ],
            'clean': True,
        }


    def pack_features(self, data_type):
        feature_path = self.workspace / 'features' / 'spectrogram' / data_type
        features = list(feature_path.rglob("*.p"))  # Search for all .p files

        return {
            'name': f'pack_features:{data_type}',
            'file_dep': features + get_source_files("utils"),
            'task_dep': [f'prepare_data:calculate_mixture_features:{data_type}'],
            'targets': [ # Targets are any feature path with data_type in it
                    x for x in self.packed_feature_paths if data_type in str(x)
                ],
            'actions': [
                Interactive(
                    f"python prepare_data.py pack_features "
                    f"--workspace={self.workspace} "
                    f"--data_type={data_type} "
                    f"--snr={self.config[f'{data_type}_snr']} "
                    f"--n_concat={self.config['n_concat']} "
                    f"--n_hop={self.config['n_hop']} "
                )
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    def write_out_scaler(self, data_type):
        snr = self.config[f'{data_type}_snr']
        return {
            'name': f'write_out_scaler:{data_type}',
            'file_dep':  self.packed_feature_paths + get_source_files("utils"),
            'task_dep':  [f'prepare_data:pack_features:{data_type}'],
            'targets': [
                self.scalar_path
            ],
            'actions': [
                Interactive(
                    f"python prepare_data.py write_out_scaler "
                    f"--workspace={self.workspace} "
                    f"--data_type={data_type} "
                    f"--snr={snr}"
                )
            ],
            'uptodate': [config_changed(self.config)],
            'clean': True,
        }


    def tasks(self):
        ''' Mix audio and extract features '''
        for task in [self.create_mixture_csv, 
                        self.calculate_mixture_features,
                        self.pack_features]:
            for data_type in ["test", "train"]:
                yield task(data_type)

        # Write out scaler is only used for train
        yield self.write_out_scaler('train')
                     
