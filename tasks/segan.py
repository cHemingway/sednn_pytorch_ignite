''' 
Pydoit tasks for SEGAN+

Will be imported from dodo.py
'''

import pathlib
import json

from doit import get_var, create_after
from doit.tools import (Interactive, PythonInteractiveAction, config_changed,
                        create_folder, run_once, title_with_actions)
from doit.task import DelayedLoader

from tasks.utils import delete_dir, get_source_files
import tasks.evaluate 

# Config specific to SEGAN only
SEGAN_CONFIG = {
     # Root folder of segan installation
    "path":   pathlib.Path(get_var('segan_path', '/home/chris/repos/segan_pytorch')),
    # HACK: Hardcode path for python EXE needed for SEGAN
    # _should_ use conda run instead, but breaks for some reason, not sure why
    'python': "/home/chris/anaconda3/envs/segan_pytorch/bin/python",
    'num_samples': get_var('segan_samples',None),
    'batch_size': get_var('segan_batch_size',100),
    'patience': get_var('segan_patience', 25) # Quit after 25 epochs if no result
}


class SEGAN_task_creator(object):
    ''' Task creator object for SEGAN '''
    def __init__(self, data, workspace_dir: pathlib.Path, 
                 result_dir: pathlib.Path, fulldata: bool, 
                 train_snr:float, test_snr:float):
        self.data = data
        self.workspace = workspace_dir
        self.result_dir = result_dir
        self.fulldata = fulldata
        self.train_snr = train_snr
        self.test_snr = test_snr
        
        # Create subdirectory names for SEGAN
        self.enhanced_dir = self.workspace / "synth_segan"
        self.train_dir = self.workspace / "segan_train_data"
        self.validation_dir = self.workspace / "segan_validation_data"
        self.tmp_dir = self.workspace / "segan_tmp"
        self.ckpt_dir = self.result_dir / "ckpt_segan+"


    @staticmethod
    def get_checkpoint(ckpt_dir: pathlib.Path) -> pathlib.Path:
        ''' Get the most recent checkpoint file for the generator '''
        log_file = ckpt_dir / 'EOE_G-checkpoints' # Create path to log file
        with open(log_file, 'r') as f: # Open it and parse the data
            try:
                log = json.load(f)
                current_file = log['current']
            except json.JSONDecodeError as e:
                raise ValueError("File does not seem to be valid JSON") from e
            except KeyError:
                raise ValueError("Could not find current checkpoint in file")
        return ckpt_dir / ('weights_' + current_file) # Add 'weights_' prefix


    def prepare_data(self):
        return {
            'name': 'prepare_data',
            'task_dep': ['calculate_mixture_features'],
            'targets': [self.train_dir, self.validation_dir],
            'actions': [
                f"rm -rf {self.tmp_dir}", # Or else SEGAN cache's old data
                Interactive(
                "python ./prepare_segan_data.py "
                f"--clean_dir={self.data['train']['speech']} "
                f"--noisy_dir={self.data['mixed']}/train/{self.train_snr}db "
                f"--train_dir={self.train_dir} "
                f"--validation_dir={self.validation_dir} "
            )]
        }


    def train(self):
        ''' Train SEGAN+ on the same testset, keeping temp files in workspace '''
        if self.fulldata:
            save_freq = 10
            epochs = 100
        else:
            save_freq = 500
            epochs = 10

        # For SEGAN, the way you specify "all samples", is by 
        # _not_ specifying --max-samples. Hence we have to do this logic
        if SEGAN_CONFIG['num_samples'] != None:
            samples_config_str  = f"--max_samples={SEGAN_CONFIG['num_samples']} "
        else:
            samples_config_str = " "


        return {
            'name':'train',
            'file_dep': get_source_files(SEGAN_CONFIG['path']),  # Depend on SEGAN source files
            'task_dep': ['segan:prepare_data'], # Depend on data task, not actual files
            'targets': [self.ckpt_dir/'EOE_D-checkpoints', 
                        self.ckpt_dir/'EOE_G-checkpoints',
                        self.ckpt_dir/'train.opts'], # TODO add .ckpt file itself
            'title': title_with_actions,
            'actions': [Interactive(
                f"time -p " # Get training time
                f"{SEGAN_CONFIG['python']} -u {SEGAN_CONFIG['path']/'train.py'} "
                f"--save_path {self.ckpt_dir} "
                f"--save_freq {save_freq} "
                f"--clean_trainset {self.train_dir}/clean "
                f"--noisy_trainset {self.train_dir}/noisy "
                f"--clean_valset {self.validation_dir}/clean "
                f"--noisy_valset {self.validation_dir}/noisy "
                f"--cache_dir {self.tmp_dir} "
                f"--epoch {epochs} "
                f"--patience {SEGAN_CONFIG['patience']} "
                f"--no_train_gen "
                f"--batch_size {SEGAN_CONFIG['batch_size']} " +
                samples_config_str +
                f"--no_bias "
                f"--slice_workers=4" # Use multiple workers
            )],
            'uptodate': [config_changed(SEGAN_CONFIG)],
            'clean': [(delete_dir, [self.train_dir])]
        }


    @create_after("segan:train",target_regex="*.wav")
    def inference(self):
        mixed = list(self.data['mixed'].rglob('*.wav'))
        try:
            segan_latest_ckpt = self.get_checkpoint(self.ckpt_dir)
        except FileNotFoundError: 
            # File is probably not found as we don't have training data yet
            # Despite using create_after, this can happen when we call doit clean
            # The action should not actually be called, so its OK
            segan_latest_ckpt = "ERROR NOT TRAINED"
        return {
            'name':'inference',
            'file_dep': mixed + get_source_files(SEGAN_CONFIG['path']) + [
                self.ckpt_dir/'EOE_G-checkpoints',
                self.ckpt_dir/'train.opts'
                ],
            'targets': [
                self.enhanced_dir,
            ],
            'title': title_with_actions, # Show full command line
            'actions': [
            (create_folder, [self.enhanced_dir]),
            Interactive(
                f"{SEGAN_CONFIG['python']} -u {SEGAN_CONFIG['path']/'clean.py'} "
                f"--g_pretrained_ckpt {segan_latest_ckpt} "
                f"--test_files {self.workspace/ 'mixed_audios/test' / str(self.test_snr)}db "
                f"--cfg_file {self.ckpt_dir/'train.opts'} "
                f"--synthesis_path {self.enhanced_dir} " #TODO move into workspace
                f"--soundfile" # Use libsoundfile backend to save
            )],
            'uptodate': [config_changed(SEGAN_CONFIG)],
        }

    def calculate_pesq(self):
        ''' Wrapper around calculate_pesq with correct parameters '''
        task = tasks.evaluate.calculate_pesq(self.workspace, 
                                             self.result_dir/"segan_pesq_results.txt",
                                             self.data, self.enhanced_dir,
                                             self.test_snr)
        task['name'] = 'calculate_pesq'
        task['task_dep'] = ['segan:inference'] # TODO depend on WAVS instead
        return task


    def calculate_bss_stoi(self):
        ''' Wrapper around calculate_bss_stoi with correct parameters '''
        task = tasks.evaluate.calculate_bss_stoi(
                                             self.result_dir/"segan_bss_stoi.csv",
                                             self.data,
                                             self.enhanced_dir)
        task['name'] = 'calculate_bss_stoi'
        task['task_dep'] = ['segan:inference'] # TODO depend on WAVS instead
        return task


    def tasks(self):
        ''' SEGAN+ GAN Based Speech Enhancement '''
        # This docstring is what pydoit shows as the "group" of tasks
        for task in [self.prepare_data, self.train, self.inference,
                     self.calculate_pesq, self.calculate_bss_stoi]:
            yield task()
