import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils')) # TODO find better way

import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from utilities import create_folder, load_hdf5
from models import DNN
import config

from audio_datasets import NoisySpeechFeaturesDataset

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

    # Paths
    train_hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'data.h5')
        
    test_hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'test', '%ddb' % int(te_snr), 'data.h5')

    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'scaler.p')
        
    models_dir = os.path.join(workspace, 'models', '{}db'.format(int(tr_snr)))

    # Datasets & dataloader
    print("Scaling data...")
    train_dataset = NoisySpeechFeaturesDataset(train_hdf5_path, scaler_path)
    test_dataset = NoisySpeechFeaturesDataset(train_hdf5_path, scaler_path)

    # Model
    data_prop = train_dataset.get_properties()
    model = DNN(**data_prop)
    
    # Data loader
    # TODO: setting num_workers > 1 breaks CUDA? Needs restart to fix!
    # See https://github.com/pytorch/pytorch/issues/2517
    print("Loading data...")
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
    trainer = create_supervised_trainer(model, optimizer, loss)
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
    tb_logger.attach(trainer,
                 log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                 event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(trainer,
                 log_handler=OptimizerParamsHandler(optimizer),
                 event_name=Events.ITERATION_STARTED)

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
        print("Interrupted, exiting..")

    # We need to close the logger with we are done
    tb_logger.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)

    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')