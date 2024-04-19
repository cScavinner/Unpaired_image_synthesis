from os.path import expanduser
from posix import listdir
from numpy import NaN, dtype
from torchio.data import image
from torchio.utils import check_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse
from model import mDRITpp
from options.train_options import TrainOptions
from load_data import load_data
home = expanduser("~")

if __name__ == '__main__':
    print("Nombre de GPUs detectes: "+ str(torch.cuda.device_count()))
    args = TrainOptions().parse()

    if args.seed:
        print('Set seed: '+str(args.seed_value))
        seed_everything(args.seed_value, workers=True)

    num_epochs = args.epochs
    gpu = args.gpu
    num_workers = args.workers
    network=args.network
    sampler=args.sampler
    training_batch_size = args.batch_size
    validation_batch_size = args.batch_size
    patch_size = args.patch_size
    samples_per_volume = args.samples
    max_queue_length = args.queue
    
    data = args.data 
    
    prefix = network+'_'
    if args.experiment_name is not None:
        prefix += args.experiment_name + '_'
    prefix += data
    prefix += '_epochs_'+str(num_epochs)
    prefix += '_patches_'+str(patch_size)

    if args.model is not None:
        prefix += '_using_init'
    
    output_path = args.saving_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    subjects=[]

    #############################################################################################################################################################################""
    # DATASET
    subjects, check_subjects = load_data(data=data, segmentation=args.segmentation, batch_size=training_batch_size)
    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(check_subjects), 'subjects')
    prefix += '_subj_'+str(len(check_subjects))+'_images_'+str(len(subjects))

    #############################################################################################################################################################################""
    # DATA AUGMENTATION
    flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
    bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
    noise = tio.RandomNoise(std=0.1, p=0.25)
    normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    spatial = tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0, p=0.75)
    
    if args.method == 'ddpm':
        normalization = tio.RescaleIntensity()

    transforms = [flip, spatial, bias, normalization, noise]
    training_transform = tio.Compose(transforms)
    validation_transform = tio.Compose([normalization])   

    #############################################################################################################################################################################""
    # TRAIN AND VALIDATION SETS
    training_sub=['sub_E05', 'sub_T03', 'sub_T02', 'sub_T05', 'sub_T01', 'sub_T04', 'sub_E02', 'sub_E01', 'sub_T08', 'sub_E08', 'sub_E06']
    validation_sub=['sub_T06']
    test_sub=['sub_E03']

    training_subjects=[]
    validation_subjects=[]
    train=[]
    validation=[]
    test=[]
    test_subjects=[]
    for s in subjects:
        if s.subject_name in training_sub:
            training_subjects.append(s)
            if s.subject_name not in train:
                train.append(s.subject_name)
        elif s.subject_name in validation_sub:
            validation_subjects.append(s)
            if s.subject_name not in validation:
                validation.append(s.subject_name)
        elif (test_sub is not None):
            if s.subject_name in test_sub:
                test_subjects.append(s)
                if s.subject_name not in test:
                    test.append(s.subject_name)
        else:
            sys.exit("Problème de construction ensembles de train et validation")
    print('training = '+str(train))
    print('validation = '+str(validation))
    print('test = '+str(test))

    training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

    test_set = tio.SubjectsDataset(
    test_subjects, transform=validation_transform)


    print('Training set:', len(training_sub), 'subjects')
    print('Validation set:', len(validation_sub), 'subjects') 
    if data!='dhcp_2mm' and data!='dhcp_1mm' and data!='dhcp_original' and data!='dhcp_1mm_npair':
        prefix=prefix+'_validation_'+str(validation)


    #############################################################################################################################################################################""
    # PATCHES SETS
    print('num_workers : '+str(num_workers))
    prefix += '_sampler_'+sampler
    if sampler=='Probabilities':
        probabilities = {0: 0, 1: 1}
        sampler = tio.data.LabelSampler(
            patch_size=patch_size,
            label_name='label',
            label_probabilities=probabilities
        )
    elif sampler=='Uniform':
        sampler = tio.data.UniformSampler(patch_size)
    else:
        sys.exit('Select a correct sampler')
    
    if args.seed:
        patches_training_set = tio.Queue(
            subjects_dataset=training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
            start_background=True,
        )

        patches_validation_set = tio.Queue(
            subjects_dataset=validation_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
            start_background=True,
        )
    else:
        patches_training_set = tio.Queue(
            subjects_dataset=training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
            start_background=True,
        )

        patches_validation_set = tio.Queue(
            subjects_dataset=validation_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
            start_background=True,
        )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size, num_workers=0, pin_memory=False)
    print("Number of train patches: " + str(len(training_loader_patches.dataset)))
    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size, num_workers=0, pin_memory=False)
    print("Number of validation patches: " + str(len(validation_loader_patches.dataset)))
  
    #############################################################################################################################################################################""
    if args.subcommand == "DRIT":
        if (args.method != 'ddpm' and args.model is None):
            net = DRIT(
                prefix = prefix,
                opt = args,
                isTrain=True
            )
        elif (args.method != 'ddpm' and args.model is not None):
            net = DRIT(
                opt=args,
                prefix=prefix,
                isTrain=True
            )

    if args.model is not None:
        pretrained_dict = torch.load(args.model)['state_dict']
        model_dict = net.state_dict()
        if model_dict.keys() == pretrained_dict.keys():
            net.load_state_dict(pretrained_dict)
        else:
            DICT = {k:(pretrained_dict[k] if (k in pretrained_dict.keys()) else model_dict[k]) for k in model_dict.keys()}
            net.load_state_dict(DICT)

    checkpoint_callback = ModelCheckpoint(dirpath=output_path)

    logger = TensorBoardLogger(save_dir = output_path, name = 'Test_logger',version=prefix)

    if gpu >= 0:
        device = 'gpu'
    else:
        device = 'cpu'

    n_gpus = torch.cuda.device_count()

    trainer_args = {
        'accelerator': device,
        'max_epochs' : num_epochs,
        'logger' : logger,
        'callbacks': checkpoint_callback,
        'precision': 16,
        'devices': [0]
    }

    if n_gpus > 1:
        trainer_args['strategy']='ddp'

    if args.seed:
        trainer_args['deterministic']='warn'
        trainer = pl.Trainer(**trainer_args)
    else:
        trainer = pl.Trainer(**trainer_args)

    trainer.fit(net, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(output_path+prefix+'.ckpt')
    torch.save(net.state_dict(), output_path+prefix+'_torch.pt')

    print('Finished Training')
