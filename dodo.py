# Alternative to runme.sh
# Run with doit tool: see pydoit.org

# To use full data, run doit fulldata=true

# You can also use full data, e.g. doit workspace=/path/to/workspace

# pylint: disable=wrong-import-order

import os
import pathlib
import shutil
import json

# HACK: Ideally would fix import paths instead
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

#pylint: disable=wrong-import-position
from utils import prepare_data
from doit.tools import (Interactive, PythonInteractiveAction, config_changed,
                        create_folder, run_once, title_with_actions)
from doit import get_var, create_after

CONFIG = {
    "fulldata": get_var('fulldata', None),
    "magnification": get_var('magnification', 2),
    "test_snr": get_var('te_snr', 0),
    "train_snr": get_var('train_snr', 0),
    "n_concat": get_var('n_concat', 7),
    "n_hop":     get_var('n_hop', 3),
}

# Use different default workspace for full data
CONFIG["workspace"] = pathlib.Path(
    get_var('workspace', 
            "workspace_full" if CONFIG["fulldata"] else "workspace")
)

RESULT_DIR = pathlib.Path(get_var("result_dir","results"))

# Use different doit database for full and partial data
DOIT_CONFIG = {
    'dep_file': '.doit{}.db'.format('_fulldata' if CONFIG['fulldata'] else '')
}

# Keep backend out of CONFIG so can calculate without needing new features
BACKEND = get_var('backend', "pytorch")

ITERATION = get_var('iteration', 10000)

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
        }
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

# Config specific to SEGAN only
SEGAN_CONFIG = {
     # Root folder of segan installation
    "path":   pathlib.Path(get_var('segan_path', '/home/chris/repos/segan_pytorch')),
    # HACK: Hardcode path for python EXE needed for SEGAN
    # _should_ use conda run instead, but breaks for some reason, not sure why
    'python': "/home/chris/anaconda3/envs/segan_pytorch/bin/python",
    'num_samples': get_var('segan_samples',None),
    'batch_size': get_var('segan_batch_size',100)
}

# Cannot use "+" in folder due to PESQ limitations
SEGAN_OUTPUT_FOLDER = CONFIG["workspace"] / "synth_segan"
SEGAN_TRAIN_FOLDER = CONFIG['workspace'] / "segan_train_data"
SEGAN_TMP_FOLDER = CONFIG['workspace'] / "segan_tmp"
# FIXME CKPT_DIR in wrong place! Should be RESULT_DIR/minidata/ckpt_segan+
SEGAN_CKPT_DIR = RESULT_DIR/"ckpt_segan+"


# Set tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Shared dependencies/targets

def get_data_filenames(data, types=('test', 'train')):
    filenames = []
    for data_type in types:
        for folder in data[data_type].values():
            folder_path = pathlib.Path(folder)
            wavfiles = folder_path.rglob("*.wav")
            filenames += list(wavfiles)
            # TODO: Must be better way of making case sensitive
            wavfiles = folder_path.rglob("*.WAV")
            filenames += list(wavfiles)
    return filenames


DATA_FILES = get_data_filenames(DATA)


PACKED_FEATURE_DIR = CONFIG["workspace"] / 'packed_features' / 'spectrogram'
PACKED_FEATURE_SUFFIX = pathlib.PurePath(
    f'{CONFIG["train_snr"]}db') / "data.h5"
PACKED_FEATURE_PATHS = [
    PACKED_FEATURE_DIR / "test" / PACKED_FEATURE_SUFFIX,
    PACKED_FEATURE_DIR / "train" / PACKED_FEATURE_SUFFIX
]

SCALAR_PATH = CONFIG["workspace"] / "packed_features" / "spectrogram" / "train" \
    / f'{CONFIG["train_snr"]}db' / "scaler.p"

MODEL_PATH = CONFIG["workspace"] / 'models' / f'{CONFIG["train_snr"]}db' \
    / f'chkpoint__ig_model_10.pth'


MIXED_WAVS_DIR = CONFIG['workspace']/"mixed_audios"

ENH_WAVS_DIR = CONFIG['workspace']/"enh_wavs"/"test" \
    / "{}db".format(CONFIG['test_snr'])

print(MODEL_PATH)

# Needed to get around the args nonsense


class DictAttr(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

#
# Wrapper functions to pack args
#


def mix_csv(subdata, data_type, magnification=1):
    args = {
        "workspace": CONFIG["workspace"],
        "speech_dir": subdata["speech"],
        "noise_dir": subdata["noise"],
        "data_type": data_type,
        "magnification": magnification
    }
    prepare_data.create_mixture_csv(DictAttr(args))


def mix_features(subdata, data_type, snr):
    args = {
        "workspace": CONFIG["workspace"],
        "speech_dir": subdata["speech"],
        "noise_dir": subdata["noise"],
        "data_type": data_type,
        "snr": snr
    }
    prepare_data.calculate_mixture_features(DictAttr(args))


def pack_features(data_type, snr, n_concat, n_hop):
    args = {
        "workspace": CONFIG["workspace"],
        "data_type": data_type,
        "snr": snr,
        "n_concat": n_concat,
        "n_hop": n_hop
    }
    prepare_data.pack_features(DictAttr(args))


def write_out_scalar(data_type, snr):
    args = {
        "workspace": CONFIG["workspace"],
        "data_type": data_type,
        "snr": snr,
        "print_scalar": False
    }
    prepare_data.write_out_scaler(DictAttr(args))

#
# Utility functions to get specific files
#


def get_source_files(folder):
    ''' Recursively get all python source files below folder '''
    folder_path = pathlib.Path(folder)
    return list(folder_path.rglob("*.py"))


def delete_workspace():
    ''' Utility function to delete workspace at end '''
    try:
        shutil.rmtree(CONFIG["workspace"]) 
    except FileNotFoundError:
        pass # Already been deleted

def delete_segan_train():
    ''' Utility function to delete workspace at end '''
    for f in [SEGAN_CKPT_DIR, SEGAN_TMP_FOLDER]:
        try:
            shutil.rmtree(f) 
        except FileNotFoundError:
            pass # Already been deleted

def segan_get_checkpoint(ckpt_dir: pathlib.Path) -> pathlib.Path:
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

#
# The actual tasks themselves
#
def task_make_workspace():
    ''' Create workspace folder if needed '''
    return {
        'targets': [CONFIG["workspace"]],
        'uptodate': [run_once],
        'actions': [
            (create_folder, [CONFIG["workspace"]]),
            (create_folder, [SEGAN_OUTPUT_FOLDER])
        ],
        'clean':[delete_workspace],
        'uptodate': [config_changed(str(CONFIG["workspace"]))],
        }


def task_create_mixture_csv():
    return {
        'file_dep':  DATA_FILES + get_source_files("utils"),
        # Using pathlib slash '/' operator
        'targets': [
            CONFIG["workspace"] / 'mixture_csvs' / 'test.csv',
            CONFIG["workspace"] / 'mixture_csvs' / 'train.csv'
        ],
        # Call mix_csv on each type of data
        # Need to wrap in PythonInteractiveAction for TQDM to work
        'actions': [
            PythonInteractiveAction(
                mix_csv, [DATA["train"], "train", CONFIG["magnification"]]),
            PythonInteractiveAction(
                mix_csv, [DATA["test"], "test", CONFIG["magnification"]]),
        ],
        'uptodate': [config_changed(CONFIG)],
        'clean': True,
    }


def task_calculate_mixture_features():
    return {
        'file_dep':  DATA_FILES + get_source_files("utils") + task_create_mixture_csv()['targets'],
        'targets': [
            MIXED_WAVS_DIR,
            CONFIG["workspace"] / "features",
        ],
        'actions': [
            PythonInteractiveAction(
                mix_features, [DATA["train"], "train", CONFIG["train_snr"]]),
            PythonInteractiveAction(
                mix_features, [DATA["test"], "test", CONFIG["test_snr"]]),
        ],
        'uptodate': [config_changed(CONFIG)],
        'clean': True,
    }


@create_after(executed='calculate_mixture_features', target_regex='.*\data.h5')
def task_pack_features():
    shared_args = [CONFIG["n_concat"], CONFIG["n_hop"]]

    feature_path = CONFIG["workspace"] / 'features'
    features = list(feature_path.rglob("*.p"))  # Search for all .p files

    return {
        'file_dep': features + get_source_files("utils"), 
        'targets': PACKED_FEATURE_PATHS,
        'actions': [
            PythonInteractiveAction(
                pack_features, ["train", CONFIG["train_snr"], *shared_args]),
            PythonInteractiveAction(
                pack_features, ["test", CONFIG["test_snr"], *shared_args]),
        ],
        'uptodate': [config_changed(CONFIG)],
        'clean': True,
    }


def task_write_out_scalar():
    return {
        'file_dep':  PACKED_FEATURE_PATHS + get_source_files("utils"),
        'targets': [
            SCALAR_PATH
        ],
        'actions': [
            PythonInteractiveAction(
                write_out_scalar, ["train", CONFIG["train_snr"]]),
        ],
        'uptodate': [config_changed(CONFIG)],
        'clean': True,
    }

@create_after(executed='calculate_mixture_features',target_regex="*.wav")
def task_prepare_segan_data():
    return {
        'file_dep' : list(pathlib.Path(MIXED_WAVS_DIR).glob("*.wav")),
        'targets': [SEGAN_TRAIN_FOLDER],
        'actions': [Interactive(
            "python ./prepare_segan_data.py "
            f"--clean_dir={DATA['train']['speech']} "
            f"--noisy_dir={MIXED_WAVS_DIR}/train/{CONFIG['train_snr']}db "
            f"--output_dir=%(targets)s "
        )]
    }


def task_train():
    return {
        'file_dep': [SCALAR_PATH] + get_source_files(BACKEND) + PACKED_FEATURE_PATHS,
        'targets': [
            MODEL_PATH
        ],
        'actions': [Interactive(
            f"python {BACKEND}/main_ignite.py train "
            f"--workspace={CONFIG['workspace']} "
            f"--tr_snr={CONFIG['train_snr']} --te_snr={CONFIG['test_snr']}"
        )],
    }


@create_after(executed='prepare_segan_data',target_regex=".*.ckpt .*.opts *checkpoints")
def task_train_segan():
    ''' Train SEGAN+ on the same testset, keeping temp files in workspace '''
    if CONFIG["fulldata"]:
        save_freq = 50
    else:
        save_freq = 500

    # For SEGAN, the way you specify "all samples", is by 
    # _not_ specifying --max-samples. Hence we have to do this logic
    if SEGAN_CONFIG['num_samples'] != None:
        samples_config_str  = f"--max_samples={SEGAN_CONFIG['num_samples']} "
    else:
        samples_config_str = " "

    return {
        'file_dep': list(SEGAN_TRAIN_FOLDER.rglob("*.wav")), # TODO depend on SEGAN code
        'targets': [SEGAN_CKPT_DIR/'EOE_D-checkpoints', 
                    SEGAN_CKPT_DIR/'EOE_G-checkpoints',
                    SEGAN_CKPT_DIR/'train.opts'], # TODO add .ckpt file itself
        'title': title_with_actions,
        'actions': [Interactive(
            f"{SEGAN_CONFIG['python']} -u {SEGAN_CONFIG['path']/'train.py'} "
            f"--save_path {SEGAN_CKPT_DIR} "
            f"--save_freq {save_freq} "
            f"--clean_trainset {DATA['train']['speech']} "
            f"--noisy_trainset {SEGAN_TRAIN_FOLDER} "
            f"--cache_dir {SEGAN_TMP_FOLDER} "
            f"--no_train_gen "
            f"--batch_size {SEGAN_CONFIG['batch_size']} " +
            samples_config_str +
            f"--no_bias "
            f"--slice_workers=4" # Use multiple workers
        )],
        'uptodate': [config_changed(SEGAN_CONFIG)],
        'clean': [delete_segan_train]
    }

@create_after(executed='calculate_mixture_features', target_regex='*.wav')
def task_inference():
    mixed = list(MIXED_WAVS_DIR.rglob('*.wav'))
    return {
        'file_dep': mixed + get_source_files(BACKEND), # TODO Add checkpoint
        'targets': [
            ENH_WAVS_DIR
        ],
        'actions': [Interactive(
            f"python {BACKEND}/main_ignite.py inference "
            f"--workspace={CONFIG['workspace']} "
            f"--tr_snr={CONFIG['train_snr']} --te_snr={CONFIG['test_snr']} "
            f"--n_concat={CONFIG['n_concat']}"
        )]
    }


@create_after(executed='train_segan', target_regex='*.wav')
def task_segan_inference():
    mixed = list(MIXED_WAVS_DIR.rglob('*.wav'))
    segan_latest_ckpt = segan_get_checkpoint(SEGAN_CKPT_DIR)
    return {
        'file_dep': mixed + get_source_files(SEGAN_CONFIG['path']) + [
            SEGAN_CKPT_DIR/'EOE_G-checkpoints',
            SEGAN_CKPT_DIR/'train.opts'
            ],
        'targets': [
            SEGAN_OUTPUT_FOLDER,
        ],
        'title': title_with_actions, # Show full command line
        'actions': [Interactive(
            f"{SEGAN_CONFIG['python']} -u {SEGAN_CONFIG['path']/'clean.py'} "
            f"--g_pretrained_ckpt {segan_latest_ckpt} "
            f"--test_files {CONFIG['workspace']/ 'mixed_audios/test' / str(CONFIG['test_snr'])}db "
            f"--cfg_file {SEGAN_CKPT_DIR/'train.opts'} "
            f"--synthesis_path {SEGAN_OUTPUT_FOLDER} " #TODO move into workspace
            f"--soundfile" # Use libsoundfile backend to save
        )],
        'uptodate': [config_changed(SEGAN_CONFIG)],
    }


@create_after(executed='inference')
def task_calculate_pesq():
    ''' Calculate PESQ of all enhanced speech '''
    return {
        'file_dep': list(ENH_WAVS_DIR.rglob("*.enh.wav")) + list(SEGAN_OUTPUT_FOLDER.rglob("*.wav")),
        'targets': [f'{RESULT_DIR}/dnn_pesq_results.txt', f'{RESULT_DIR}/segan_pesq_results.txt'],
        'actions': [
            # Evaluate PESQ
            Interactive(
                f"python evaluate_pesq.py calculate_pesq "
                f"--workspace={CONFIG['workspace']} "
                f"--speech_dir={DATA['test']['speech']} --te_snr={CONFIG['test_snr']} "
            ),
            f"mv _pesq_results.txt {RESULT_DIR}/dnn_pesq_results.txt",
            # Evaluate SEGAN
            Interactive(
                f"python evaluate_pesq.py calculate_pesq "
                f"--workspace={CONFIG['workspace']} "
                f"--speech_dir={DATA['test']['speech']} "
                f"--enh_speech_dir={SEGAN_OUTPUT_FOLDER} "
                f"--te_snr={CONFIG['test_snr']} "
            ),
            f"mv _pesq_results.txt {RESULT_DIR}/segan_pesq_results.txt",
            # Cleanup
            "rm _pesq_itu_results.txt"
        ],
    }

@create_after(executed='inference')
def task_calculate_bss_stoi():
    ''' Calculate bss and stoi of all enhanced speech '''
    return {
        'file_dep': list(ENH_WAVS_DIR.rglob("*.enh.wav")) + list(SEGAN_OUTPUT_FOLDER.rglob("*.wav")),
        'targets': [f'{RESULT_DIR}/dnn_bss_stoi.csv', f'{RESULT_DIR}/segan_bss_stoi.csv'],
        'actions': [
            # Evaluate PESQ
            Interactive(
                f"python evaluator.py "
                f"-q " # Hide warnings
                f"--clean_dir={DATA['test']['speech']} "
                f"--dirty_dir={ENH_WAVS_DIR} "
                f"--output_file={RESULT_DIR}/dnn_bss_stoi.csv"
            ),
            # Evaluate SEGAN
            Interactive(
                f"python evaluator.py "
                f"-q "
                f"--clean_dir={DATA['test']['speech']} "
                f"--dirty_dir={SEGAN_OUTPUT_FOLDER} "
                f"--output_file={RESULT_DIR}/segan_bss_stoi.csv"
            ),
        ],
    }


def task_plot():
    ''' Generate plots for all data'''
    return {
        'file_dep': [f'{RESULT_DIR}/dnn_pesq_results.txt', f'{RESULT_DIR}/segan_pesq_results.txt',
                    f'{RESULT_DIR}/dnn_bss_stoi.csv', f'{RESULT_DIR}/segan_bss_stoi.csv'],
        'targets': [f'{RESULT_DIR}/segan_plot.png', f'{RESULT_DIR}/dnn_plot.png'],
        'actions': [
            f"python show_stats.py --csv_file={RESULT_DIR}/dnn_bss_stoi.csv "
                f"--pesq_file={RESULT_DIR}/dnn_pesq_results.txt "
                f"--plot_file={RESULT_DIR}/dnn_plot.png",
            f"python show_stats.py --csv_file={RESULT_DIR}/segan_bss_stoi.csv "
                f"--pesq_file={RESULT_DIR}/segan_pesq_results.txt "
                f"--plot_file={RESULT_DIR}/segan_plot.png"
        ],
    }

def task_backup_results():
    ''' Save results into .tar.gz with current date/time whenever changed '''
    return {
        'file_dep': task_calculate_bss_stoi()['targets'] +
                    task_calculate_pesq()['targets'] +
                    task_plot()['targets'],
        'targets': [f'{RESULT_DIR}/previous'],
        'actions': [
            # Remove older SEGAN checkpoints to save ~1GB of disk!
            f"python {SEGAN_CONFIG['path']}/purge_ckpts.py {SEGAN_CKPT_DIR}", 
            # Backup everything else in results dir
            f"bash backup_results.sh {RESULT_DIR}"
        ]
    }



def task_get_stats():
    ''' Calculate overall stats '''
    return {
        'file_dep': [f'{RESULT_DIR}/dnn_pesq_results.txt', f'{RESULT_DIR}/segan_pesq_results.txt',
                    f'{RESULT_DIR}/dnn_bss_stoi.csv', f'{RESULT_DIR}/segan_bss_stoi.csv'],
        'actions': [
            Interactive("echo DNN ------------------"),
            Interactive(f"python show_stats.py --csv_file={RESULT_DIR}/dnn_bss_stoi.csv --pesq_file={RESULT_DIR}/dnn_pesq_results.txt"),
            Interactive("echo SEGAN+  -------------"),
            Interactive(f"python show_stats.py --csv_file={RESULT_DIR}/segan_bss_stoi.csv --pesq_file={RESULT_DIR}/segan_pesq_results.txt"),
        ],
        
        'uptodate': [False]  # Always run this
    }
