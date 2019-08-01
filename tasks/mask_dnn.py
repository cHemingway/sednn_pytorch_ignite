''' Vanilla mask based dnn '''

import pathlib

from doit import get_var, create_after
from doit.tools import Interactive


from tasks.utils import get_source_files
import tasks.evaluate

# Keep backend out of CONFIG so can calculate without needing new features
BACKEND = "pytorch"


class MASK_DNN_basic_creator(object):
    ''' Task creator object for existing mask based DNN '''

    def __init__(self, data, workspace_dir: pathlib.Path,
                 result_dir: pathlib.Path,
                 scalar_path, packed_feature_paths,
                 fulldata: bool,
                 train_snr: float, test_snr: float, n_concat: int):

        self.data = data
        self.workspace = workspace_dir
        self.scalar_path = scalar_path
        self.packed_feature_paths = packed_feature_paths
        self.result_dir = result_dir
        self.fulldata = fulldata
        self.train_snr = train_snr
        self.test_snr = test_snr
        self.n_concat = n_concat

        self.model_path = self.workspace / 'models' / f'{self.train_snr}db' \
            / f'chkpoint__ig_model_10.pth'

        self.enhanced_dir = self.workspace/"enh_wavs" / \
            "test" / "{}db".format(self.test_snr)

    def train(self):
        return {
            'name': 'train',
            'file_dep': [self.scalar_path] + get_source_files(BACKEND) + self.packed_feature_paths,
            'targets': [
                self.model_path
            ],
            'actions': [Interactive(
                f"python {BACKEND}/main_ignite.py train "
                f"--workspace={self.workspace} "
                f"--tr_snr={self.train_snr} --te_snr={self.test_snr}"
            )],
        }

    @create_after(executed='calculate_mixture_features', target_regex='*.wav')
    def inference(self):
        mixed = list(self.data['mixed'].rglob('*.wav'))
        return {
            # TODO Add checkpoint
            'name': 'inference',
            'file_dep': mixed + get_source_files(BACKEND) + [self.model_path],
            'targets': [
                self.enhanced_dir
            ],
            'actions': [Interactive(
                f"python {BACKEND}/main_ignite.py inference "
                f"--workspace={self.workspace} "
                f"--tr_snr={self.train_snr} --te_snr={self.test_snr} "
                f"--n_concat={self.n_concat}"
            )]
        }


    @create_after("mask_basic_dnn:inference",target_regex="*.*")
    def calculate_pesq(self):
        ''' Wrapper around calculate_pesq with correct parameters '''
        task = tasks.evaluate.calculate_pesq(self.workspace,
                                                self.result_dir/"mask_basic_dnn_pesq_results.txt",
                                                self.data, self.enhanced_dir,
                                                self.test_snr)
        task['name'] = 'calculate_pesq'
        clean_wav_files = list(self.enhanced_dir.rglob("*.wav"))
        task['file_dep'] = clean_wav_files
        return task


    @create_after("mask_basic_dnn:inference",target_regex="*.*")
    def calculate_bss_stoi(self):
        ''' Wrapper around calculate_bss_stoi with correct parameters '''
        task = tasks.evaluate.calculate_bss_stoi(
            self.result_dir/"mask_basic_dnn_bss_stoi.csv",
            self.data,
            self.enhanced_dir)
        task['name'] = 'calculate_bss_stoi'
        # TODO depend on WAVS instead
        clean_wav_files = list(self.enhanced_dir.rglob("*.wav"))
        task['file_dep'] = clean_wav_files
        return task


    def tasks(self):
        ''' Original IRM Mask and Log Mag  '''
        # This docstring is what pydoit shows as the "group" of tasks
        for task in [self.train, self.inference,
                    self.calculate_pesq, self.calculate_bss_stoi]:
            yield task()
