# Main file to run all experiments
# Cannot be ran directly! Run this with doit tool: see pydoit.org

# To use full data, run doit fulldata=true

# You can also use full data, e.g. doit workspace=/path/to/workspace

import os
import pathlib
import json


from doit.tools import (Interactive, PythonInteractiveAction, config_changed,
                        create_folder, run_once, title_with_actions)
from doit import get_var, create_after


# Import other tasks and utilities
from tasks.segan import SEGAN_task_creator
from tasks.mask_dnn import MASK_DNN_basic_creator 
from tasks.data_prepare import Data_Prepare_creator
from tasks.utils import get_data_filenames, get_source_files, delete_dir,\
                         delete_dirs, copy_sample_files

CONFIG = {
    "fulldata": get_var('fulldata', None),
    "magnification": get_var('magnification', 2),
    "test_snr": get_var('te_snr', 0),
    "train_snr": get_var('train_snr', 0),
    "n_concat": get_var('n_concat', 7),
    "n_hop":     get_var('n_hop', 3),
    "extra_speakers": get_var('extra_speakers',1),
    "extra_speech_db": get_var('extra_speech_db',-5)
}

# Use different default workspace for full data
CONFIG["workspace"] = pathlib.Path(
    get_var('workspace', 
            "workspace_full" if CONFIG["fulldata"] else "workspace")
)

RESULT_DIR = pathlib.Path(get_var("result_dir","/mnt/Spare/Project/results"))

# Use different doit database for full and partial data
DOIT_CONFIG = {
    'dep_file': '.doit{}.db'.format('_fulldata' if CONFIG['fulldata'] else '')
}


DATA = {}
if CONFIG["fulldata"]:
    DATA = {
        "train": {
            "speech": "metadata/train_speech",
            "noise":  "metadata/train_noise",
        },
        "test": {
            "speech": "metadata/test_speech",
            "noise":  "metadata/test_noise",
        },
    }
    RESULT_DIR = RESULT_DIR / "metadata"
else:
    DATA = {
        "train": {
            "speech": "mini_data/train_speech",
            "noise":  "mini_data/train_noise",
        },
        "test": {
            "speech": "mini_data/test_speech",
            "noise":  "mini_data/test_noise",
        }
    }
    RESULT_DIR = RESULT_DIR / "mini_data"

DATA["mixed"] =  CONFIG['workspace']/"mixed_audios"

PACKED_FEATURE_DIR = CONFIG["workspace"] / 'packed_features' / 'spectrogram'
PACKED_FEATURE_PATHS = [
    PACKED_FEATURE_DIR / "test" /  f'{CONFIG["test_snr"]}db' / "data.h5",
    PACKED_FEATURE_DIR / "train" /  f'{CONFIG["train_snr"]}db' / "data.h5"
]

SCALAR_PATH = CONFIG["workspace"] / "packed_features" / "spectrogram" / "train" \
    / f'{CONFIG["train_snr"]}db' / "scaler.p"


# Set tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

#
# The actual tasks themselves
#
def task_make_workspace():
    ''' Create workspace folder if needed '''
    return {
        'targets': [CONFIG["workspace"]],
        'actions': [
            (create_folder, [CONFIG["workspace"]])
        ],
        'clean':[(delete_dir, [CONFIG["workspace"]] )],
        'uptodate': [config_changed(str(CONFIG["workspace"]))],
        }


def task_prepare_data():
    task_gen = Data_Prepare_creator(DATA, CONFIG['workspace'],
                                     SCALAR_PATH, PACKED_FEATURE_PATHS, CONFIG)
    yield task_gen.tasks()


BASIC_ENHANCED_DIR = CONFIG['workspace'] /"enh_wavs" / "DNN"
@create_after(executed='prepare_data', target_regex='*.*')
def task_mask_basic_dnn():
    task_gen = MASK_DNN_basic_creator(DATA, CONFIG['workspace'], 
                                     "DNN",
                                     BASIC_ENHANCED_DIR,
                                     RESULT_DIR,
                                     SCALAR_PATH, PACKED_FEATURE_PATHS,
                                     CONFIG['fulldata'],
                                     CONFIG['train_snr'], CONFIG['test_snr'],
                                     CONFIG['n_concat'])
    yield task_gen.tasks()


LSTM_ENHANCED_DIR = CONFIG['workspace'] /"enh_wavs" / "LSTM"
@create_after(executed='prepare_data', target_regex='*.*')
def task_mask_lstm():
    task_gen = MASK_DNN_basic_creator(DATA, CONFIG['workspace'],
                                        "LSTM",
                                        LSTM_ENHANCED_DIR,
                                        RESULT_DIR,
                                        SCALAR_PATH, PACKED_FEATURE_PATHS,
                                        CONFIG['fulldata'],
                                        CONFIG['train_snr'], CONFIG['test_snr'],
                                        CONFIG['n_concat'])
    yield task_gen.tasks()


SEGAN_ENHANCED_DIR = CONFIG['workspace'] / "enh_wavs"/"synth_segan" # Files cleaned by SEGAN

# @create_after(executed='prepare_data', target_regex='*.*')
# def task_segan():
#     segan_task_gen = SEGAN_task_creator(DATA, CONFIG['workspace'], 
#                                         SEGAN_ENHANCED_DIR,
#                                         RESULT_DIR,
#                                         CONFIG['fulldata'],
#                                         CONFIG['train_snr'], CONFIG['test_snr'])
#     yield segan_task_gen.tasks()



def task_plot():
    ''' Plot everything we have data for '''
    
    def models_from_files(results, suffix):
        ''' Find the name of all models with given file suffixes '''
        files = results.glob("*"+suffix)
        # Converts "path/to/modelname_pesq.txt" into "modelname"
        return [str(f.name).replace(suffix,"") for f in files]

    pesq_models = models_from_files(RESULT_DIR, "_pesq_results.txt")
    bss_models = models_from_files(RESULT_DIR, "_bss_stoi.csv")

    # Only plot what we have _both_ pesq and bss files for
    # TODO skip part of plot if data not available?
    models = set.union(set(pesq_models),set(bss_models))

    for model in models:
        yield {
            'name': model,
            'file_dep' : [f'{RESULT_DIR}/{model}_pesq_results.txt', f'{RESULT_DIR}/{model}_bss_stoi.csv'],
            'targets': [f'{RESULT_DIR}/{model}_plot.png'],
            'actions': [
                f"python show_stats.py --csv_file={RESULT_DIR}/{model}_bss_stoi.csv "
                    f"--pesq_file={RESULT_DIR}/{model}_pesq_results.txt "
                    f"--plot_file={RESULT_DIR}/{model}_plot.png"
                    f" > {RESULT_DIR}/{model}_summary.txt" # Save summary to text file as well
            ],
        }


@create_after(executed='plot')
def task_backup_results():
    ''' Save results into .tar.gz with current date/time whenever changed '''
    NUM_WAVS_BACKUP = get_var('wavs_backup', 10)  # Backup 10 clean/noisy files
    MIXED_WAVS_TEST = DATA['mixed'] / 'test'/ f"{CONFIG['test_snr']}db/"

    # TODO get backup_list automatically from contents of enhanced_dir
    backup_list = [
        ('segan', SEGAN_ENHANCED_DIR),
        ('basic', BASIC_ENHANCED_DIR),
        ('lstm',LSTM_ENHANCED_DIR)
    ]

    # Copy sample files from enhanced dir to sample_dir, saving deps produced
    backup_task_deps = []
    for model_name, enh_dir in backup_list:
        sample_dir = RESULT_DIR/f'{model_name}_sample'
        sample_task_name = f'sample_{model_name}'
        # Hack, to keep a list of these dependencies, we need to generate the 
        # same name for this task that pydoit will use. This way, we can ensure
        # backup_dir depends on all of the tasks we generate here
        backup_task_deps.append('backup_results:' + sample_task_name)
        yield {
            'name' : sample_task_name,
            'task_dep': ['plot'],
            'targets': [sample_dir],
            'actions': [
                (copy_sample_files,
                    [ MIXED_WAVS_TEST, enh_dir, sample_dir, NUM_WAVS_BACKUP ]
                )
            ]
        }

    # Backup the folder
    yield {
        'name': 'backup_dir',
        'task_dep' : backup_task_deps,
        'targets': [f'{RESULT_DIR}/previous'],
        'actions': [
            # Backup everything else in results dir
            f"bash scripts/backup_results.sh {RESULT_DIR}"
        ]
    }