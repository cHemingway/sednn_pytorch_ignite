import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils')) # TODO find better way

import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler, OptimizerParamsHandler, WeightsHistHandler)

from termcolor import colored
from tqdm import tqdm

from utilities import (create_folder, load_hdf5, scale, np_mean_absolute_error, 
    pad_with_border, log_sp, mat_2d_to_3d, inverse_scale, get_stft_window_func, 
    write_audio, read_audio, calculate_spectrogram)
from models import DNN, move_data_to_gpu
import config
from stft import real_to_complex, istft, get_cola_constant, overlap_add


from audio_datasets import NoisySpeechFeaturesDataset


def plot_spectrogram(x,y,n_concat):
    ''' Given Tensors x,y, returns a figure of the first 1000 items '''
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].matshow(x[0 : 1000, n_concat // 2, :].numpy().T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(y[0 : 1000, :].numpy().T, origin='lower', aspect='auto', cmap='jet')
    axs[0].set_title('Mixture')
    axs[1].set_title('Clean')
    plt.tight_layout()
    return fig


def train(args):

    # Arugments & parameters
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    batch_size = 500
    lr = 1e-4
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print(colored('No GPU Found!','red'))

    # Paths
    train_hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'data.h5')
        
    test_hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'test', '%ddb' % int(te_snr), 'data.h5')

    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'scaler.p')
        
    models_dir = os.path.join(workspace, 'models', '{}db'.format(int(tr_snr)))

    # Datasets & dataloader
    print("Loading Dataset, Scaling data...")
    train_dataset = NoisySpeechFeaturesDataset(train_hdf5_path, scaler_path)
    test_dataset = NoisySpeechFeaturesDataset(train_hdf5_path, scaler_path)

    # Model
    data_prop = train_dataset.get_properties()
    model = DNN(**data_prop)

    if torch.cuda.device_count() > 1:
        print(colored('Using {} GPUs'.format(torch.cuda.device_count()),'green'))
        model = torch.nn.DataParallel(model)
    
    # Data loader
    # TODO: setting num_workers > 1 breaks CUDA? Needs restart to fix!
    # See https://github.com/pytorch/pytorch/issues/2517
    print("Dataloader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function, mean average error (L1)
    loss = F.l1_loss

    # Setup trainer and evalduator
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            device=device,
                                            metrics={
                                                # Accuracy() # TODO find replacement for accuracy
                                                'loss': Loss(loss)
                                                })

    # TODO make difference between training and test loss clear
    
    # Progress bar attach
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, ['loss'])

    # Tensorboard attach, training loss and optimizer params
    tb_logger = TensorboardLogger(os.path.join(workspace, 'tensorboard'))

    # Plot spectrogram before/after cleaning
    n_concat = train_dataset.get_properties()['n_concat']
    fig = plot_spectrogram(train_dataset.x,train_dataset.y,n_concat)
    tb_logger.writer.add_figure('Spectrogram',fig,global_step=0,close=True)

    # Attach tensorboard loggers
    tb_logger.attach(trainer,
                 log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                 event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(trainer,
                 log_handler=OptimizerParamsHandler(optimizer),
                 event_name=Events.ITERATION_STARTED)

    tb_logger.attach(trainer,
                log_handler=WeightsHistHandler(model),
                event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        ''' Show test results every epoch '''
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        pbar.log_message("Test Set Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(trainer.state.epoch, 0, metrics['loss']))

    # Handler to save checkpoints
    chkpoint_handler = ModelCheckpoint(models_dir, 
                                        filename_prefix="chkpoint_", 
                                        save_interval=2, # Save every 2nd epoch
                                        require_empty=False # Overwrite
                                        )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, chkpoint_handler, 
                                {
                                    'ig_model': model,
                                    'ig_optimizer': optimizer    
                                })
    
    print("Starting training")
    try:
        trainer.run(train_loader, max_epochs=10)
    except KeyboardInterrupt:
        print(colored("Interrupted, exiting..",'red'))
    else:
        # Print done
        print(colored('FINISHED','green'))
    finally:
        # We need to close the logger with we are done
        tb_logger.close()


def inference(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
    """
    # This function is from original sednn codebase, lightly modified
    
    # Arguments & parameters
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    data_type = 'test'
    
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size - overlap
    cuda = torch.cuda.is_available()
    freq_bins = window_size // 2 + 1
    sample_rate = config.sample_rate
    window = get_stft_window_func(config.window_type)(window_size)
    
    # Paths
    mixed_audios_dir = os.path.join(workspace, 'mixed_audios', data_type, 
        '{}db'.format(int(te_snr)))
        
    features_dir = os.path.join(workspace, 'features', 'spectrogram', data_type, 
        '{}db'.format(int(te_snr)))
    
    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'scaler.p')
    
    model_path = os.path.join(workspace, 'models', '{}db'.format(int(tr_snr)), 
        'chkpoint__ig_model_10.pth')
        
    enh_audios_dir = os.path.join(workspace, 'enh_wavs', data_type, 
        '{}db'.format(int(te_snr)))

    # Load model
    model = DNN(n_concat, freq_bins)   
    state_dict = torch.load(model_path) 

    # Hack to remove results of DataParallel
    # See https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    # TODO detect if required, e.g. might be trained on only 1 GPU
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    # Convert model to CUDA, but single threaded
    if cuda:
        model.cuda()
    
    # Load scaler
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    feature_names = os.listdir(features_dir)

    # Create output folder
    create_folder(enh_audios_dir)

    for audio_name in tqdm(feature_names):
    
        # Load feature
        feature_path = os.path.join(features_dir, audio_name)
        data = pickle.load(open(feature_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        
        mixed_x = np.abs(mixed_cmplx_x)
        
        # Process data
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        mixed_x = log_sp(mixed_x)
        speech_x = log_sp(speech_x)
        
        # Scale data
        mixed_x = scale(mixed_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        
        # Move data to GPU
        mixed_x_3d = move_data_to_gpu(mixed_x_3d, cuda)
        
        # Predict
        prediction = model(mixed_x_3d)
        prediction = prediction.data.cpu().numpy()
        
        # Inverse scale
        mixed_x = inverse_scale(mixed_x, scaler)
        prediction = inverse_scale(prediction, scaler)
        
        # Recover enhanced wav
        prediction_sp = np.exp(prediction)
        complex_sp = real_to_complex(prediction_sp, mixed_cmplx_x)
        frames = istft(complex_sp)
        
        # Overlap add
        cola_constant = get_cola_constant(hop_size, window)
        enh_audio = overlap_add(frames, hop_size, cola_constant)
        
        # Write out enhanced wav
        bare_name = os.path.splitext(audio_name)[0]
        out_path = os.path.join(enh_audios_dir, '{}.enh.wav'.format(bare_name))
        write_audio(out_path, enh_audio, sample_rate)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)

    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        raise Exception('Error argument!')