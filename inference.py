from tqdm import tqdm
import os
import numpy as np
import glob
from torchio.data.image import ScalarImage
home = expanduser("~")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import math
import argparse
from options.test_options import TestOptions
from model import mDRITpp


if __name__ == '__main__':
    args = TestOptions().parse()
    # Create the result folder
    if not os.path.exists(os.path.join(args.model, 'images')):
        os.mkdir(os.path.join(args.model, 'images'))
    record_path = os.path.join(args.model, 'images')

    # Set the device
    if args.gpu is not None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")
    else:
        device = torch.device("cpu")
    gpu = args.gpu
    network = args.network
    

    # Initialization of the network 
    if args.subcommand == "DRIT":
        net = mDRITpp(
            prefix = '',
            opt = args,
            isTrain=isTrain
        )
        

    # Loading the model
    model = glob.glob(os.path.join(args.model, '*.pt'))[0]
    if model.split('/')[-1].split('.')[-1]=='pt':
        net.load_state_dict(torch.load(model))
    elif model.split('/')[-1].split('.')[-1]=='ckpt':
        net.load_state_dict(torch.load(model)['state_dict'])
    else:
        sys.exit('Set a valid checkpoint')
    net.eval()
    if args.gpu is not None:
        net.to(device=device)

    
    # Create the subject 
    subject = tio.Subject(
        LR_image=tio.ScalarImage(args.input),
        HR_image=tio.ScalarImage(args.ground_truth),
        )

    
    # Data Normalization
    normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    augment = normalization
    sub = augment(subject)
    

    # Create grid Sampler and grid aggregator
    patch_size = (128, 128, 1)
    patch_overlap = (96, 96, 0)
    grid_sampler = tio.inference.GridSampler(
        sub,
        patch_size,
        patch_overlap
        )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    
    if args.mode == 'reconstruction':
        aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
    elif args.mode == 'degradation':
        aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
    elif args.mode == 'both':
        aggregator_HR = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
        aggregator_LR = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

    
    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):
            LR_tensor = patches_batch['LR_image'][tio.DATA]
            if gpu is not None:
                LR_tensor=LR_tensor.to(device)
            LR_tensor=LR_tensor.squeeze(-1)
            
            if gpu is not None:
                HR_tensor = patches_batch['HR_image'][tio.DATA]
                HR_tensor=HR_tensor.to(device)
                HR_tensor=HR_tensor.squeeze(-1)
                if args.use_segmentation_network == True:
                    fake_HR, fake_LR, segmentation_HR, segmentation_LR = net(LR_tensor, HR_tensor) 
                    fake_HR = fake_HR.cpu().detach()
                    fake_LR = fake_LR.cpu().detach()
                    segmentation_HR = segmentation_HR.cpu().detach()
                    segmentation_LR = segmentation_LR.cpu().detach()
                else:
                    if args.subcommand == "DRIT":
                        if args.mode == 'reconstruction' or args.mode=='degradation':
                            outputs = net(LR_tensor, HR_tensor).cpu().detach()
                        elif args.mode == 'both':
                            outputs_HR, outputs_LR = net(LR_tensor, HR_tensor)
                            outputs_HR = outputs_HR.cpu().detach()
                            outputs_LR = outputs_LR.cpu().detach()
                    else:
                        outputs = net(LR_tensor, HR_tensor).cpu().detach()
            else:
                HR_tensor = patches_batch['HR_image'][tio.DATA]
                HR_tensor=HR_tensor.squeeze(-1)
                if args.use_segmentation_network == True:
                    fake_HR, fake_LR, segmentation_HR, segmentation_LR = net(LR_tensor, HR_tensor) 
                    fake_HR = fake_HR.detach()
                    fake_LR = fake_LR.detach()
                    segmentation_HR = segmentation_HR.detach()
                    segmentation_LR = segmentation_LR.detach()
                else:
                    if args.subcommand == "DRIT":
                        if args.mode == 'reconstruction' or args.mode=='degradation':
                            outputs = net(LR_tensor, HR_tensor).detach()
                        elif args.mode == 'both':
                            outputs_HR, outputs_LR = net(LR_tensor, HR_tensor)
                            outputs_HR = outputs_HR.detach()
                            outputs_LR = outputs_LR.detach()
                    else:
                        outputs = net(LR_tensor, HR_tensor).detach()


            if args.use_segmentation_network == True:
                segmentation_HR = segmentation_HR.unsqueeze(-1)
                segmentation_LR = segmentation_LR.unsqueeze(-1)
                fake_LR = fake_LR.unsqueeze(-1)
                fake_HR = fake_HR.unsqueeze(-1)
                aggregator_HR.add_batch(fake_HR, locations)
                aggregator_LR.add_batch(fake_LR, locations)
                aggregator_segHR.add_batch(segmentation_HR, locations)
                aggregator_segLR.add_batch(segmentation_LR, locations)
            elif args.mode == 'reconstruction' or args.mode == 'degradation':
                outputs = outputs.unsqueeze(-1)
                aggregator.add_batch(outputs, locations)
            elif args.mode == 'both':
                outputs_HR = outputs_HR.unsqueeze(-1)
                outputs_LR = outputs_LR.unsqueeze(-1)
                aggregator_LR.add_batch(outputs_LR, locations)
                aggregator_HR.add_batch(outputs_HR, locations)

    if args.use_segmentation_network == True:
        output_HR = aggregator_HR.get_output_tensor()
        output_LR = aggregator_LR.get_output_tensor()
        output_segLR = aggregator_segLR.get_output_tensor()
        output_segHR = aggregator_segHR.get_output_tensor()
        output_segLR = nn.Softmax(dim=0)(output_segLR)
        output_segHR = nn.Softmax(dim=0)(output_segHR)
        output_segHR = torch.argmax(output_segHR, dim=0).unsqueeze(0)
        output_segLR = torch.argmax(output_segLR, dim=0).unsqueeze(0)
    elif args.mode == 'reconstruction' or args.mode == 'degradation':
        output_tensor = aggregator.get_output_tensor()
    elif args.mode == 'both':
        output_fake_HR = aggregator_HR.get_output_tensor()
        output_fake_LR = aggregator_LR.get_output_tensor()
        
    else:
        print('Saving images')
        if args.use_segmentation_network:
            image_HR = tio.ScalarImage(tensor=output_HR.to(torch.float), affine=subject['LR_image'].affine)
            image_LR = tio.ScalarImage(tensor=output_LR.to(torch.float), affine=subject['HR_image'].affine)
            image_segmentation_HR = tio.LabelMap(tensor=output_segHR.to(torch.int), affine=subject['HR_image'].affine)
            image_segmentation_LR = tio.LabelMap(tensor=output_segLR.to(torch.int), affine=subject['LR_image'].affine)
            image_HR.save(os.path.join(args.output, 'Disentangled_reconstruction_'+args.ground_truth.split('/')[-1]))
            image_LR.save(os.path.join(args.output, 'Disentangled_degradation_'+args.ground_truth.split('/')[-1]))
            image_segmentation_HR.save(os.path.join(args.output, 'Disentangled_segmentation_HR_'+args.ground_truth.split('/')[-1]))
            image_segmentation_LR.save(os.path.join(args.output, 'Disentangled_segmentation_LR_'+args.ground_truth.split('/')[-1]))
        elif args.mode == 'reconstruction':
            output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['LR_image'].affine)
            output_seg.save(os.path.join(args.output, 'Disentangled_reconstruction_'+args.ground_truth.split('/')[-1]))
        elif args.mode == 'degradation':
            output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['HR_image'].affine)
            output_seg.save(os.path.join(args.output, 'Disentangled_degradation_'+args.ground_truth.split('/')[-1]))
        elif args.mode == 'both':
            output_fake_HR = tio.ScalarImage(tensor=output_fake_HR.to(torch.float), affine=subject['LR_image'].affine)
            output_fake_LR = tio.ScalarImage(tensor=output_fake_LR.to(torch.float), affine=subject['HR_image'].affine)
            output_fake_HR.save(os.path.join(args.output, 'Disentangled_reconstruction_'+args.ground_truth.split('/')[-1]))
            output_fake_LR.save(os.path.join(args.output, 'Disentangled_degradation_'+args.ground_truth.split('/')[-1]))
    

    



    
    
  
