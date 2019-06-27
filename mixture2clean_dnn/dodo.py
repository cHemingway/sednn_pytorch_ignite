## Alternative to runme.sh
# Run with doit tool: see pydoit.org

# To use full data, run doit fulldata=true

# You can also use full data, e.g. doit workspace=/path/to/workspace

import pathlib
import os

# HACK: Ideally would fix import paths instead
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

from utils.prepare_data import create_mixture_csv

from doit import get_var
from doit.tools import create_folder, PythonInteractiveAction

config = {
    "fulldata": get_var('fulldata', None),
    "workspace": pathlib.Path(get_var('workspace', "workspace")),
    "magnification": get_var('magnification', 2)
}


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

def task_make_workspace(): 
    return {
            'actions': [
                (create_folder, [config["workspace"]])
            ]}

# Needed to get around the args nonsense
class DictAttr(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

def mix_csv(subdata, data_type, magnification=1):
    args = {
        "workspace":config["workspace"],
        "speech_dir":subdata["speech"],
        "noise_dir":subdata["noise"],
        "data_type":data_type,
        "magnification":magnification
    }
    create_mixture_csv(DictAttr(args))

def get_source_files(folder):
    ''' Recursively get all python source files below folder '''
    folder_path = pathlib.Path(folder)
    return list(folder_path.rglob("*.py"))

def get_data_filenames(data):
    filenames = []
    for data_type in ['test','train']:
        for folder in data[data_type].values():
            folder_path = pathlib.Path(folder)
            wavfiles = folder_path.rglob("*.wav")
            filenames += list(wavfiles)
             # TODO: Must be better way of making case sensitive
            wavfiles = folder_path.rglob("*.WAV")
            filenames += list(wavfiles)
    return filenames

def task_create_mixture_csv():
    return {
        'file_dep' : get_data_filenames(data) + get_source_files("utils"),
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
        'clean': True,
    }
    

