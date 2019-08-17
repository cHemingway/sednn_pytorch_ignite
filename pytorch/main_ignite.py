import os, sys
# Add parent of parent folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import pickle
import inspect
import pathlib
from datetime import datetime

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

from utils.utilities import (create_folder, load_hdf5, scale, np_mean_absolute_error, 
    pad_with_border, log_sp, mat_2d_to_3d, inverse_scale, get_stft_window_func, 
    write_audio, read_audio, calculate_spectrogram, load_features, Range)
import models
import utils.config as config
from utils.stft import real_to_complex, istft, get_cola_constant, overlap_add

from audio_datasets import NoisySpeechFeaturesDataset
import wandb


def get_models():
    ''' Get dict 'name':value of all models by inspecting models.py '''
    # is_module uses short circuiting to check if it is a class _before_ checking
    # if it is a subclass, as issubclass throws exception if passed something that
    # is _not_ a class
    is_module = lambda x : inspect.isclass(x) and issubclass(x,torch.nn.Module)
    # Returns [(key,value), (key,value)], so turn into dict
    models_list =  inspect.getmembers(models,predicate=is_module)
    return dict(models_list)

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
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    device = 'cpu'
    validate_every = 100

    # Initialise W&B, ensuring we remove the tricky "model" attribute from args
    safe_config = vars(args).copy() # Convert to dictionary and get a copy
    safe_config.pop('model') # Remove model attribute
    safe_config.pop('mode') # Remove needless mode attribute, will always be train
    safe_config.pop('loss') # Remove loss attribute
    wandb.init(project="mss_speech_sep_lstm",config=safe_config)

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

    # Save model to cloud when done
    wandb.save(str(models_dir) + '*.pth')

    # Datasets & dataloader
    print("Loading Dataset, Scaling data...")
    train_dataset = NoisySpeechFeaturesDataset(train_hdf5_path, scaler_path)
    test_dataset = NoisySpeechFeaturesDataset(train_hdf5_path, scaler_path)

    # Get properties embedded in dataset, e.g. n_concat        
    data_prop = train_dataset.get_properties()

    # Split train_dataset into train and validation set
    train_len = len(train_dataset)
    validation_len = int(args.validation_split * train_len)
    train_len = train_len - validation_len
    train_dataset, validation_dataset = torch.utils.data.random_split(
                                                        train_dataset, 
                                                        (train_len, validation_len))
    print("Split dataset, {} train {} validation".format(train_len, validation_len))


    # Load Model and pass in properties of dataset
    print(f"Using Model {args.model_name}")
    model = args.model(**data_prop, dropout=dropout)

    if torch.cuda.device_count() > 1:
        print(colored('Using {} GPUs'.format(torch.cuda.device_count()),'green'))
        model = torch.nn.DataParallel(model).to(device)
    
    # Data loader
    # TODO: setting num_workers > 1 breaks CUDA? Needs restart to fix!
    # See https://github.com/pytorch/pytorch/issues/2517
    print("Dataloader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            pin_memory=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function, specified by args
    loss = args.loss

    # Setup trainer and evaluator
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
    date_str = datetime.now().strftime("%H:%M:%S %d:%m:%y")
    tb_logger = TensorboardLogger(os.path.join(workspace, f'tensorboard/{args.model_name}/{date_str}'))

    # Add graph to tensorboard
    # FIXME Doesnt work
    # freq_bins = config.window_size // 2 + 1
    # tb_logger.writer.add_graph(model, torch.zeros(5,5,freq_bins), True)

    # Plot spectrogram before/after cleaning
    # TODO plot after!
    # n_concat = data_prop['n_concat']
    # fig = plot_spectrogram(train_dataset.x,train_dataset.y,n_concat)
    # tb_logger.writer.add_figure('Spectrogram',fig,global_step=0,close=True)

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


    @trainer.on(Events.ITERATION_COMPLETED)
    def validate(trainer):
        ''' Show validation_loss results every validate_every '''
        if trainer.state.iteration % validate_every == 0:
            evaluator.run(validation_loader)
            metrics = evaluator.state.metrics
            # Log to tensorboard
            tb_logger.writer.add_scalar('training/val_loss', metrics['loss'], 
                                        global_step=trainer.state.iteration)
            # Log to w&b
            wandb.log({"Validation Loss": metrics['loss']}, step=trainer.state.iteration)

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        ''' Show test results every epoch '''
        state = evaluator.run(test_loader)
        #pylint: disable=no-member
        metrics = state.metrics
        loss = metrics['loss']
        # Show in progress bar
        pbar.log_message("Test Set Results - Epoch: {}  Avg loss: {:.2f}"
            .format(trainer.state.epoch, loss))
        # Show in train set
        wandb.log({"Test Loss": loss}, step=trainer.state.iteration)

    # Handler to save checkpoints
    chkpoint_handler = ModelCheckpoint(models_dir, 
                                        filename_prefix=f"{args.model_name}_chkpoint_", 
                                        save_interval=2, # Save every 2nd epoch
                                        require_empty=False # Overwrite
                                        )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, chkpoint_handler, 
                                {
                                    'ig_model': model,
                                    'ig_optimizer': optimizer    
                                })

    # Watch model with W&B, logging both gradients and parameters
    wandb.watch(model, log='all')
    
    print(f"Starting training {args.model_name}")
    try:
        trainer.run(train_loader, max_epochs=args.max_epochs)
    except KeyboardInterrupt:
        print(colored("Interrupted, exiting..",'red'))
    else:
        # Print done
        print(colored('FINISHED','green'))
    finally:
        # We need to close the logger with we are done
        tb_logger.close()

    # TODO run inference and upload PESQ, STOI results


def infer_test_audio(device, model, features_dir, audio_name, n_concat, scaler):
    ''' Run inference on an audio file using model '''

    window_size = config.window_size
    freq_bins = window_size // 2 + 1
    window = get_stft_window_func(config.window_type)(window_size)
    overlap = config.overlap
    hop_size = window_size - overlap

    # Load features
    feature_path = os.path.join(features_dir, audio_name)
    data = load_features(feature_path)
    mixed_complx_x, speech_x, noise_x, alpha, mrcg = data

    # Convert complex input x to real
    mixed_x = np.abs(mixed_complx_x)
    
    # Pad and log data
    n_pad = (n_concat - 1) // 2
    mixed_x = pad_with_border(mixed_x, n_pad)
    mixed_x = log_sp(mixed_x)
    speech_x = log_sp(speech_x)
    
    # Scale data
    mixed_x = scale(mixed_x, scaler)
    
    # Cut input spectrogram to 3D segments with n_concat
    mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
    
    # Covert data to torch.Tensor
    mixed_x_3d = torch.Tensor(mixed_x_3d).to(device)
    
    # Predict
    prediction = model(mixed_x_3d)

    # Move back to CPU
    prediction = prediction.data.cpu().numpy()
    
    # Inverse scale
    mixed_x = inverse_scale(mixed_x, scaler)
    prediction = inverse_scale(prediction, scaler)
    
    # Recover enhanced wav
    prediction_sp = np.exp(prediction)
    complex_sp = real_to_complex(prediction_sp, mixed_complx_x)
    frames = istft(complex_sp)
    
    # Overlap add
    cola_constant = get_cola_constant(hop_size, window)
    enh_audio = overlap_add(frames, hop_size, cola_constant)
    
    return enh_audio


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_rate = config.sample_rate
    window_size = config.window_size
    freq_bins = window_size // 2 + 1

    
    # Paths
    mixed_audios_dir = os.path.join(workspace, 'mixed_audios', data_type, 
        '{}db'.format(int(te_snr)))
        
    features_dir = os.path.join(workspace, 'features', 'spectrogram', data_type, 
        '{}db'.format(int(te_snr)))
    
    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'scaler.p')
    

    # TODO model path changes when number of epochs change
    model_path = os.path.join(workspace, 'models', '{}db'.format(int(tr_snr)), 
        f'{args.model_name}_chkpoint__ig_model_10.pth')
        
    enh_audios_dir = args.enhanced_dir

    # Load model
    model = args.model(n_concat, freq_bins)   
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
    
    # Move model to device if needed
    model.to(device)
    
    # Load scaler
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Get list of audio names
    audio_names = os.listdir(features_dir)

    # Create output folder
    create_folder(enh_audios_dir)

    for name in tqdm(audio_names):
        # Infer the audio
        enh_audio = infer_test_audio(device, model, features_dir, name, n_concat, scaler)

        # Write out enhanced wav
        bare_name = os.path.splitext(name)[0]
        out_path = os.path.join(enh_audios_dir, '{}.enh.wav'.format(bare_name))
        write_audio(out_path, enh_audio, sample_rate)

if __name__ == '__main__':

    models = get_models() # Get dict of models in models.py

    loss_functions = {
        'L1': F.l1_loss,
        'MSE': F.mse_loss
    }

    parser = argparse.ArgumentParser(description='')

    parser.add_argument("model_name", help="Model from models.py to use",
                        choices=models.keys())

    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, default=1e-5)
    parser_train.add_argument('--batch_size', type=int, default=1000)
    parser_train.add_argument('--max_epochs', type=int, default=10)
    parser_train.add_argument('--loss_func', dest="loss_name", 
                              choices=loss_functions, default='MSE')
    parser_train.add_argument('--validation_split', type=float, default=0.1,
                              choices=[Range(0.0, 0.2)])
    parser_train.add_argument('--dropout', type=float, default=0.2,
                              choices=[Range(0.0, 0.5)])

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--enhanced_dir', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)

    args = parser.parse_args()

    # Add model to arguments
    args.model = models[args.model_name]
    

    if args.mode == 'train':
        args.loss = loss_functions[args.loss_name]
        train(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        raise Exception('Error argument!')