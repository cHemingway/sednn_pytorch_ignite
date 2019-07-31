'''
Evaluation tasks for use by implementations
'''

import pathlib
from doit.tools import Interactive


def calculate_pesq(workspace_dir: pathlib.Path, result_file: pathlib.Path,
                   data, enhanced_dir, test_snr: float):
    ''' Calculate PESQ of all enhanced speech '''
    return {
        'targets': [result_file],
        'actions': [
            # Evaluate PESQ
            Interactive(
                f"python evaluate_pesq.py calculate_pesq "
                f"--workspace={workspace_dir} "
                f"--speech_dir={data['test']['speech']} --te_snr={test_snr} "
                f"--enh_speech_dir={enhanced_dir} "
            ),
            f"mv _pesq_results.txt {result_file}",
            # Cleanup
            "rm _pesq_itu_results.txt"
        ],
    }


def calculate_bss_stoi(result_file, data, enhanced_dir):
    ''' Calculate bss and stoi of all enhanced speech '''
    return {
        'targets': [result_file],
        'actions': [
            # Evaluate PESQ
            Interactive(
                f"python evaluator.py "
                f"-q "  # Hide warnings
                f"--clean_dir={data['test']['speech']} "
                f"--dirty_dir={enhanced_dir} "
                f"--output_file={result_file} "
            ),
        ],
    }
