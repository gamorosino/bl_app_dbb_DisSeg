#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:21:11 2021

@author: Gabriele Amorosino
"""

import argparse
from UnetBrainSeg import init_unet,unet_test

gpu_num=str(1)
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_LENGTH = 256
dims=(IMG_WIDTH, IMG_HEIGHT, IMG_LENGTH)
    
#%% Main

if __name__ == '__main__':
    
    ## Parsing Inputs
    parser = argparse.ArgumentParser(description='Predict segmentation')
    parser.add_argument('fullpath', metavar='images', type=str, nargs='+',
                        help='fullpath of T1w file file')
    parser.add_argument('fullpath1', metavar='labels', type=str, nargs='+',
                        help='fullpath of ouput segmentation file')
    parser.add_argument('fullpath2', metavar='checkpoints_dir', type=str, nargs='+',
                        help='fullpath of checkpoints directories')   
    
    args = parser.parse_args()
    TEST_PATH_x=args.fullpath[0]    
    TEST_PATH_y=args.fullpath1[0] 
    #outputfile=args.fullpath1[0]
    checkpoints_dir=args.fullpath2[0]
    
    ## initialize the U-Net
    unet=init_unet(checkpoints_dir,gpu_num=gpu_num)
    
    ## Perform prediction and save results
    unet_test(TEST_PATH_x,TEST_PATH_y, unet, dims,iamge_type='float32',label_type='float32',ncores=1)
    
