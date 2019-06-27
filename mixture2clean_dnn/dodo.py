## Alternative to runme.sh
# Run with doit tool: see pydoit.org

# To use full data, run doit fulldata=true

# You can also use full data, e.g. doit workspace=/path/to/workspace

import pathlib
import os

# HACK: Ideally would fix import paths instead
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

from utils import prepare_data

from doit import get_var
from doit.tools import \
    create_folder, Interactive, PythonInteractiveAction, config_changed, run_once

config = {
    "fulldata": get_var('fulldata', None),
    "workspace": pathlib.Path(get_var('workspace', "workspace")),
    "magnification": get_var('magnification', 2),
    "test_snr": get_var('te_snr', 0),
    "train_snr": get_var('train_snr', 0),
    "n_concat": get_var('n_concat', 7),
    "n_hop":     get_var('n_hop', 3)
}

# Keep backend out of config so can calculate without needing new features
BACKEND = get_var('backend', "pytorch")

ITERATION = get_var('iteration', 10000)

data = {}
if config["fulldata"]:
    data = {
        "train": {
            "speech": "metadata/train_speech",
            "noise":  "metadata/train_noise",
        },
        "test": {
            "speech": "metadata/test_speech",
            "noise":  "metadata/test_noise",
        }
    }
else:
    data = {
        "train": {
            "speech": "mini_data/train_speech",
            "noise":  "mini_data/train_noise",
        },
        "test": {
            "speech": "mini_data/test_speech",
            "noise":  "mini_data/test_noise",
        }
    }

# Set tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# Shared dependencies/targets

def get_data_filenames(data, types=['test','train']):
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

DATA_FILES = get_data_filenames(data)


PACKED_FEATURE_DIR = config["workspace"] / 'packed_features' / 'spectrogram'
PACKED_FEATURE_SUFFIX = pathlib.PurePath(f'{config["train_snr"]}db') / "data.h5"
PACKED_FEATURE_PATHS = [
    PACKED_FEATURE_DIR / "test" / PACKED_FEATURE_SUFFIX,
    PACKED_FEATURE_DIR / "train" / PACKED_FEATURE_SUFFIX
]

SCALAR_PATH = config["workspace"] / "packed_features" / "spectrogram" / "train" \
               / f'{config["train_snr"]}db' / "scaler.p"

MODEL_PATH =  config["workspace"] / 'models' / f'{config["train_snr"]}db' \
                / f'md_{ITERATION}_iters.tar'

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
        "workspace":config["workspace"],
        "speech_dir":subdata["speech"],
        "noise_dir":subdata["noise"],
        "data_type":data_type,
        "magnification":magnification
    }
    prepare_data.create_mixture_csv(DictAttr(args))


def mix_features(subdata, data_type, snr):
    args = {
        "workspace":config["workspace"],
        "speech_dir":subdata["speech"],
        "noise_dir":subdata["noise"],
        "data_type":data_type,
        "snr":snr
    }
    prepare_data.calculate_mixture_features(DictAttr(args))


def pack_features(data_type, snr, n_concat, n_hop):
    args = {
        "workspace":config["workspace"],
        "data_type":data_type,
        "snr":snr,
        "n_concat":n_concat,
        "n_hop":n_hop
    }
    prepare_data.pack_features(DictAttr(args))


def write_out_scalar(data_type, snr):
    args = {
        "workspace":config["workspace"],
        "data_type":data_type,
        "snr":snr,
        "print_scalar":False
    }
    prepare_data.write_out_scaler(DictAttr(args))

#
# Utility functions to get specific files
#

def get_source_files(folder):
    ''' Recursively get all python source files below folder '''
    folder_path = pathlib.Path(folder)
    return list(folder_path.rglob("*.py"))



#
# The actual tasks themselves
#
def task_make_workspace(): 
    ''' Create workspace folder if needed '''
    return {
            'targets': [ config["workspace"] ],
            'uptodate': [run_once],
            'actions': [
                (create_folder, [config["workspace"]])
            ]}


def task_create_mixture_csv():
    return {
        'file_dep' :  DATA_FILES + get_source_files("utils"),
        # Using pathlib slash '/' operator
        'targets': [
            config["workspace"] / 'mixture_csvs' / 'test.csv',
            config["workspace"] / 'mixture_csvs' / 'train.csv'
        ],
        # Call mix_csv on each type of data
        # Need to wrap in PythonInteractiveAction for TQDM to work
        'actions': [
            PythonInteractiveAction(mix_csv, [data["train"], "train", config["magnification"]] ),
            PythonInteractiveAction(mix_csv, [data["test"], "test", config["magnification"]] ),
        ],
        'uptodate': [config_changed(config)],
        'clean': True,
    }


def task_calculate_mixture_features():
    return {
        'file_dep' :  DATA_FILES + get_source_files("utils"),
        'targets' : [
            config["workspace"] / "mixed_audios",
            config["workspace"] / "features",
        ],
        'actions': [
            PythonInteractiveAction(mix_features, [data["train"], "train", config["train_snr"]] ),
            PythonInteractiveAction(mix_features, [data["test"], "test", config["test_snr"]] ),
        ],
        'uptodate': [config_changed(config)],
        'clean': True,
    }


def task_pack_features():
    shared_args = [config["n_concat"], config["n_hop"]]

    feature_path = config["workspace"] / 'features'
    features = list(feature_path.rglob("*.p")) # Search for all .p files

    return {
        'file_dep' :  features + get_source_files("utils"),
        'targets' : PACKED_FEATURE_PATHS,
        'actions': [
            PythonInteractiveAction(pack_features, ["train",  config["train_snr"], *shared_args]),
            PythonInteractiveAction(pack_features, ["test", config["test_snr"], *shared_args]),
        ],
        'uptodate': [config_changed(config)],
        'clean': True,
    }


def task_write_out_scalar():

    packed_feature_path = config["workspace"] / 'packed_features'
    packed_features = list(packed_feature_path.rglob("*.h5"))

    return {
        'file_dep' :  packed_features + get_source_files("utils"),
        'targets' : [
            SCALAR_PATH
        ],
        'actions': [
            PythonInteractiveAction(write_out_scalar, ["train",  config["train_snr"]]),
        ],
        'uptodate': [config_changed(config)],
        'clean': True,
    }

def task_train():
    return {
        'file_dep': [SCALAR_PATH] + get_source_files(BACKEND) + PACKED_FEATURE_PATHS, # TODO rest of dependencies
        'targets' : [
            MODEL_PATH
        ],
        'actions' : [Interactive(
            f"python {BACKEND}/main.py train "
            f"--workspace={config['workspace']} "
            f"--tr_snr={config['train_snr']} --te_snr={config['test_snr']}"
        )],
    }

def task_inference():
    test_data = get_data_filenames(data,['test']) # Get only the deps
    return {
        'file_dep': test_data + get_source_files(BACKEND) + [MODEL_PATH] + PACKED_FEATURE_PATHS,
        # TODO targets
        'actions' : [Interactive(
            f"python {BACKEND}/main.py inference "
            f"--workspace={config['workspace']} "
            f"--tr_snr={config['train_snr']} --te_snr={config['test_snr']} "
            f"--iteration={ITERATION} --n_concat={config['n_concat']}"
        )]
    }

def task_calculate_pesq():
    ''' Calculate PESQ of all enhanced speech '''
    return {
        # TODO: Get file_dep from inference target
        'targets': ['_pesq_results.txt', '_pesq_itu_results.txt'],
        'actions' : [Interactive(
            f"python evaluate.py calculate_pesq "
            f"--workspace={config['workspace']} "
            f"--speech_dir={data['test']['speech']} --te_snr={config['test_snr']} "
        )],
    }