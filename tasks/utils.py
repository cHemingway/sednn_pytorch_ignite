''' Utility functions shared amongst tasks '''
import pathlib
import shutil

def get_data_filenames(data, types=('test', 'train')):
    ''' Get the filenames of data in test and train directories '''
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


def get_source_files(folder):
    ''' Recursively get all python source files below folder '''
    folder_path = pathlib.Path(folder)
    return list(folder_path.rglob("*.py"))


def delete_dir(dir):
    try:
        shutil.rmtree(dir) 
    except FileNotFoundError:
        pass # Already been deleted
    

def delete_dirs(dirs):
    ''' Utility function to delete folders and contents for cleanup'''
    for f in dirs:
        delete_dir(f)
            

def copy_sample_files(noisy_dir: pathlib.Path, enhanced_dir: pathlib.Path,
                      out_dir: pathlib.Path, max_n=10) ->  None:
    ''' Copy up to max_n noisy/enhanced files to result_dir '''
    def copy_n_wavs(src,dst,max_n):
        ''' Helper function, copies n in alphabetical order '''
        wav_names = list(src.glob("*.wav"))
        wav_names.sort() # Sort in place so get same each time
        for n,f in enumerate(wav_names):
            if n>max_n:
                break
            shutil.copy(str(f),str(dst))
  
    # Dictionary names
    out_noisy = out_dir / 'noisy'
    out_enhanced = out_dir / 'enhanced'
    # Create dirs
    out_dir.mkdir(exist_ok=True)
    out_noisy.mkdir(exist_ok=True)
    out_enhanced.mkdir(exist_ok=True)
    # Copy the wavs
    copy_n_wavs(noisy_dir.absolute(), out_noisy.absolute(), max_n)
    copy_n_wavs(enhanced_dir.absolute(), out_enhanced.absolute(), max_n) 