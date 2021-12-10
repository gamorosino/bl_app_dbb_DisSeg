#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:06:37 2021

@author: Gabriele Amorosino
"""



import os
from libraries.DATAMANlib import NormData,skresize,trymakedir,integerize_seg,get_data,NormImages
from libraries.google_drive_downloader import GoogleDriveDownloader as gdd
from libraries.UNET import UNET_3D_multiclass, ConfusionMatrix
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import  binary_dilation as bDil

divis=0.5
IMG_WIDTH = int(float(128 / divis))
IMG_HEIGHT = int(float(128 / divis))
IMG_LENGTH = int(float(128 /divis ))
fltr1stnmb=12;
drop_rate=0.15
loss_type="sparse_softmax_cross_entropy"
ext_name_train="DisSeg"

#%%
def tf_resp(T1_img,dims):    
    img=NormData(T1_img);                                
    img = np.expand_dims(skresize( img ,dims, mode='constant',order=0), axis=-1);
    img = np.expand_dims(img , axis=0);
    return img


def load_nib(T1_file):    
    T1_Struct=nib.load(T1_file);
    T1_aff=T1_Struct.get_affine(); 
    T1_header=T1_Struct.get_header();
    T1_img = T1_Struct.get_data();
    return T1_img,T1_header,T1_aff

def init_unet(checkpoint_dir=None,checkpoint_basename=None,ckpt_step=None,checkpoints_dir=None,gpu_num=str(0),num_labels=7):

   
   fbname=str(IMG_WIDTH)+'x'+str(IMG_HEIGHT)+'x'+str(IMG_LENGTH)
   if checkpoint_dir is None:
       trymakedir(checkpoints_dir)
       checkpoint_dir = checkpoints_dir + "/checkpoints/"+ext_name_train+"_"+fbname+"_"+"filter"+str(fltr1stnmb)+"_dp"+str(drop_rate)+"_"+loss_type+"/"
       print(checkpoint_dir) 
   if checkpoint_basename is None:
       checkpoint_basename = ext_name_train

   if ckpt_step is None:
       ckpt_step="24202"
   checkpoint_file1=checkpoint_dir+ "/"+checkpoint_basename+"-"+ckpt_step+".data-00000-of-00001"
   checkpoint_file2=checkpoint_dir+ "/"+checkpoint_basename+"-"+ckpt_step+".index"
   checkpoint_file3=checkpoint_dir+ "/"+checkpoint_basename+"-"+ckpt_step+".meta"
   	
   if not os.path.isfile(checkpoint_file1) or not os.path.isfile(checkpoint_file2) or not os.path.isfile(checkpoint_file3):
	print("Download checkpoint...")
	
	trymakedir(checkpoint_dir)

	gdd.download_file_from_google_drive(file_id='',
                                    dest_path=checkpoint_file1)
	gdd.download_file_from_google_drive(file_id='',
                                    dest_path=checkpoint_file2)
	gdd.download_file_from_google_drive(file_id='',
                                    dest_path=checkpoint_file3)	
   checkpoint_file=checkpoint_dir+ "/checkpoint"
   
   if not os.path.isfile(checkpoint_file):
	   model_checkpoint_path="model_checkpoint_path: "+"\""+checkpoint_dir+""+checkpoint_basename+"-"+ckpt_step+"\""
	   all_model_checkpoint_paths="all_model_checkpoint_paths: "+"\""+checkpoint_dir+""+checkpoint_basename+"-"+ckpt_step+"\""	   
	   outF = open(checkpoint_file, "w")
	   for line in [model_checkpoint_path,all_model_checkpoint_paths]:
		 print >>outF, line
	   outF.close()
   print("checkpoint_dir: "+checkpoint_dir)
   unet = UNET_3D_multiclass( loss_type=loss_type,
                                          drop_rate=drop_rate,
                                          filter1stnumb=fltr1stnmb,
                                          ckpt_dir=checkpoint_dir,
                                          ckpt_basename=checkpoint_basename,
                                          ckpt_step=ckpt_step,gpu_num=gpu_num,num_labels=num_labels)

   return unet


def unet_predict(T1_file,outputfile,unet,dims,Mask_file=None):
    #load T1
    T1_img,T1_header,T1_aff=load_nib(T1_file)
    if Mask_file is not None:
		MaskArray = nib.load(Mask_file).get_data()
		MaskArray_orig=MaskArray.copy()
		MaskArray = bDil(MaskArray, structure=None, iterations=1)
		T1_img = ( MaskArray * T1_img ) +  (  2000 * ( MaskArray == 0 ) )
		
    img=tf_resp(T1_img,dims)
    #Predict segmentation    
    predictedSeg=unet.predict(img);
    predictedSeg=skresize( predictedSeg , T1_img.shape, mode='constant',order=0);
    if mask is not None:
		predictedSeg = predictedSeg * MaskArray_orig	    
    
    #save results
    seg_T1_int_Struct = nib.Nifti1Image(integerize_seg(predictedSeg), affine=T1_aff, header=T1_header);
    seg_T1_int_Struct.to_filename(outputfile)    
    return predictedSeg


def unet_test(TEST_PATH_x,TEST_PATH_y, unet, dims,iamge_type='float32',label_type='float32',ncores=1):
    iamge_type=np.dtype(iamge_type)
    label_type=np.dtype(label_type)
    X_test, Y_test = get_data(TEST_PATH_x,TEST_PATH_y, iamge_type,label_type,dims,ncores=ncores)
    ix=range(X_test.shape[0])
    X_test=NormImages(X_test)
    segs = unet.test(X_test,
                    Y_test,
                    indices=ix,
                    plot=False,
                    plot_separately=True,
                    compute_accuracy=True,
                    compute_metrics=True,
                    plot_accuracy=False,
                    print_separately=True,
                    plot_ground_truth=False)
    #np.save(Dice_score_list_file_all,unet.dice_score)
    return segs

def dice_score(predicted_file,gtruth_file ,seg_labels=None):
    
    predicted, _, _=load_nib(predicted_file)
    gtruth, _, _=load_nib(gtruth_file)
    
    if seg_labels is None:
                  seg_labels=np.unique(predicted)
                  
                  if np.any(seg_labels==0.0):
                      seg_labels=np.delete(seg_labels, 0)
                  print("labels found: "+str(seg_labels))

    if isinstance(seg_labels, list) :
                      seg_labels.sort()
    elif isinstance(seg_labels, np.ndarray):
                      np.sort(seg_labels)
    elif isinstance(seg_labels, int):               
                       seg_labels=[seg_labels]                 
    else:
                       seg_labels=[seg_labels]
    
    Dice_score=np.zeros(len(seg_labels))
    for idx,value in enumerate(seg_labels): 
        predicted_i=(predicted==value).astype(np.int)
        gtruth_i=(gtruth==value).astype(np.int)
        ConfMat = ConfusionMatrix(predicted_i, gtruth_i)
        Dice_score[idx]=ConfMat.Dice_score
    return Dice_score


