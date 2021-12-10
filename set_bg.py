#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:20:42 2021

@author: Gabriele Amorosino
"""

import argparse
import nibabel as nib
from scipy.ndimage.morphology import  binary_dilation as bDil

if __name__ == '__main__':


    
    ## Inputs
    parser = argparse.ArgumentParser(description='Predict segmentation')
    parser.add_argument('fullpath', metavar='T1', type=str, nargs='+',
                        help='fullpath of T1w file file')
    parser.add_argument('fullpath1', metavar='mask', type=str, nargs='+',
                        help='fullpath of brain mask file')
    parser.add_argument('fullpath2', metavar='output', type=str, nargs='+',
                        help='fullpath of T1-w output file')   
        

    
    args = parser.parse_args()
    T1_file=args.fullpath[0]    
    Mask_file=args.fullpath1[0]
    T1_file_bg=args.fullpath2[0]
                        
    MaskArray = nib.load(Mask_file).get_data()
    MaskArray_orig=MaskArray.copy()
    MaskArray = bDil(MaskArray, structure=None, iterations=1)
                      
    T1_NIB = nib.load(T1_file)
    T1_Array = T1_NIB.get_data()
    T1_header = T1_NIB.get_header()
    T1_affine = T1_NIB.get_affine()
                        
    T1_bg2000 = ( MaskArray * T1_Array ) +  (  2000 * ( MaskArray == 0 ) )
                        
    T1_NIB_bg2000 = nib.Nifti1Image(T1_bg2000, affine=T1_affine,header=T1_header)
    T1_NIB_bg2000.to_filename(T1_file_bg)