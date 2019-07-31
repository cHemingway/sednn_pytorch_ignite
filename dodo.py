# Alternative to runme.sh
# Run with doit tool: see pydoit.org

# To use full data, run doit fulldata=true

# You can also use full data, e.g. doit workspace=/path/to/workspace

# pylint: disable=wrong-import-order

import os
import pathlib
import json

# HACK: Ideally would fix import paths instead
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

#pylint: disable=wrong-import-position
from utils import prepare_data
from doit.tools import (Interactive, PythonInteractiveAction, config_changed,
                        create_folder, run_once, title_with_actions)
from doit import get_var, create_after

# Import SEGAN task
from tasks.segan import SEGAN_task_creator
from tasks.utils import get_data_filenames, get_source_files, delete_dirs

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

RESULT_DIR = pathlib.Path(get_var("result_dir","/mnt/Spare/Project/results"))

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


# Set tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


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
# The actual tasks themselves
#
def task_make_workspace():
    ''' Create workspace folder if needed '''
    return {
        'targets': [CONFIG["workspace"]],
        'actions': [
            (create_folder, [CONFIG["workspace"]])
        ],
        'clean':[(delete_dirs, [CONFIG["workspace"]] )],
        'uptodate': [config_changed(str(CONFIG["workspace"]))],
        }


def task_create_mixture_csv():
    return {
        'file_dep':  DATA_FILES + get_source_files("utils"),
        'task_dep': ['make_workspace'],
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
            DATA['mixed'],
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
        'task_dep': ['calculate_mixture_features'],
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


@create_after(executed='calculate_mixture_features', target_regex='*.wav')
def task_inference():
    mixed = list(DATA['mixed'].rglob('*.wav'))
    return {
        'file_dep': mixed + get_source_files(BACKEND), # TODO Add checkpoint
        'task_dep': ['train'],
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


@create_after(executed='calculate_mixture_features', target_regex='*.wav')
def task_segan():
    segan_task_gen = SEGAN_task_creator(DATA, CONFIG['workspace'], RESULT_DIR,
                                        CONFIG['fulldata'],
                                        CONFIG['train_snr'], CONFIG['test_snr'])
    yield segan_task_gen.tasks()



def task_calculate_pesq():
    ''' Calculate PESQ of all enhanced speech '''
    return {
        'task_dep': ['inference', 'segan'],
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


def task_calculate_bss_stoi():
    ''' Calculate bss and stoi of all enhanced speech '''
    return {
        'task_dep': ['inference', 'segan'],
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
        'task_dep' : ['calculate_bss_stoi', 'calculate_pesq'],
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
    NUM_WAVS_BACKUP = get_var('wavs_backup', 10)  # Backup 10 clean/noisy files
    MIXED_WAVS_TEST = DATA['mixed'] / 'test'/ f"{CONFIG['test_snr']}db/"
    return {
        'task_dep': ['plot', 'get_stats'],
        'targets': [f'{RESULT_DIR}/previous'],
        'actions': [
            # Backup SEGAN files
            (copy_sample_files,
             [
                 MIXED_WAVS_TEST, SEGAN_OUTPUT_FOLDER,
                 RESULT_DIR/'segan_sample', NUM_WAVS_BACKUP
             ]),
            # Backup DNN files
            (copy_sample_files,
             [
                 MIXED_WAVS_TEST, ENH_WAVS_DIR,
                 RESULT_DIR/'dnn_sample', NUM_WAVS_BACKUP
             ]),
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
        'targets': [f'{RESULT_DIR}/dnn_summary.txt', f'{RESULT_DIR}/segan_summary.txt'],
        'actions': [
            Interactive("echo DNN ------------------"),
            Interactive(f"python show_stats.py --csv_file={RESULT_DIR}/dnn_bss_stoi.csv --pesq_file={RESULT_DIR}/dnn_pesq_results.txt | tee {RESULT_DIR}/dnn_summary.txt"),
            Interactive("echo SEGAN+  -------------"),
            Interactive(f"python show_stats.py --csv_file={RESULT_DIR}/segan_bss_stoi.csv --pesq_file={RESULT_DIR}/segan_pesq_results.txt | tee {RESULT_DIR}/segan_summary.txt"),
        ],
    }
