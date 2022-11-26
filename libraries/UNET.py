#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:05:49 2019


#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
################### title:                                UNET Library                                ###################
###################                                                                                   ###################
###################   description:                                                                    ###################
###################   version:       0.8.3.0.0                                                        ###################
###################   notes:	        need to Install: software                                     ###################
###################                  py modules - tf,matplotlib,nibabel,skimage,                      ###################
###################                               tqdm                                                ###################
###################                                                                                   ###################
###################   bash version:   tested on GNU bash, version 4.3.48                              ###################
###################                                                                                   ###################
################### autor: gamorosino                                                                 ###################
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
###################         Update : new class UNET_3D_light                                      
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################

@author: gamorosino
"""

import os
import numpy as np
import scipy

import random
import math
import warnings
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import rotate as skrotate
from skimage.util import random_noise as skrandom_noise
from skimage.transform import AffineTransform as skaffine
from skimage.transform import warp as skwarp
from skimage.transform import swirl as skswirl
from skimage import exposure

import tensorflow as tf
from tensorflow.python.ops import array_ops
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rc('figure', max_open_warning = 0)
from multiprocessing import Pool #  Process pool  
#personal modules
from PLOTlib import OrtoView, plot_overlay
from DATAMANlib import trymakedir,NormImages,StandData, integerize, integerize_seg 
from LossLib import dice_coe
from functools import partial
#%%
########################################################################################################################
#################################################  FUNCTIONS   #########################################################
################################ ############################################################ ########################



def close_pool(pool):
       pool.close()
       pool.terminate()
       pool.join()     



def expand_concat(X_out,X_int):
    
     return np.concatenate([X_out,np.expand_dims(X_int,axis=0)],axis=0)

def random_dilation(X,Y,factor=[0.8, 1.2]):
      scale_factor = random.uniform(factor[0],factor[1])
      afftransf=skaffine(scale=[scale_factor,scale_factor])
      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
      Y_scaled=skwarp(Y,afftransf, preserve_range=True,mode="edge", order=0)
      return X_scaled,Y_scaled


def AgumentData(X,Y,**kwargs):
      nosie_flag = kwargs.get('noise', False)
      tp=Y.dtype
      
      # Rotate
      X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
      random_degree = random.uniform(-90, 90)
      X_a=skrotate(X_scaled, random_degree, preserve_range=True,mode="edge")
      Y_a=skrotate(Y_scaled, random_degree, preserve_range=True,mode="edge", order=0)      
      Y_a=(Y_a>0).astype(tp)
      X_agu=np.expand_dims(np.copy(X_a),axis=0)
      Y_agu=np.expand_dims(np.copy(Y_a),axis=0)
      X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
      random_degree = random.uniform(-45, 45)
      X_b=(skrotate(X_scaled.transpose(),random_degree, preserve_range=True,mode="edge")).transpose()
      Y_b=(skrotate(Y_scaled.transpose(),random_degree, preserve_range=True,mode="edge", order=0)).transpose()
      Y_b=(Y_b>0).astype(tp)
      X_agu=expand_concat(X_agu,X_b)
      Y_agu=expand_concat(Y_agu,Y_b)
      
      #X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
      #random_degree = random.uniform(-45, 45)
      #X_c=skrotate(X_scaled.transpose(0,2,1),random_degree, preserve_range=True,mode="edge").transpose(0,2,1)
      #Y_c=skrotate(Y_scaled.transpose(0,2,1),random_degree, preserve_range=True,mode="edge").transpose(0,2,1)
      #Y_c=(Y_c>0).astype(tp)
      #X_agu=expand_concat(X_agu,X_c)
      #Y_agu=expand_concat(Y_agu,Y_c)
      
      # Flip
      #X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
      #X_hflip=X_scaled[:, ::-1,:]
      #Y_hflip=Y_scaled[:, ::-1,:]
      #Y_hflip=(Y_hflip>0).astype(tp)
      #X_agu=expand_concat(Y_agu,X_hflip)
      #Y_agu=expand_concat(Y_agu,Y_hflip)
      #X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
      #X_vflip=X_scaled[::-1, :,:]
      #Y_vflip=Y_scaled[::-1, :,:]
      #Y_vflip=(Y_vflip>0).astype(tp)
      #X_agu=expand_concat(X_agu,X_vflip)
      #Y_agu=expand_concat(Y_agu,Y_vflip)
      # Affine
      scale_factor = random.uniform(0.8,1.2)
      random_degree = random.uniform(-0.2, 0.2)
      afftransf=skaffine(shear=random_degree, scale=[scale_factor,scale_factor])
      X_affine=skwarp(X,afftransf, preserve_range=True,mode="edge")
      Y_affine=skwarp(Y,afftransf, preserve_range=True,mode="edge", order=0)
      Y_affine=(Y_affine>0).astype(tp)
      X_agu=expand_concat(X_agu,X_affine)
      Y_agu=expand_concat(Y_agu,Y_affine)
      # Swirl
      X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])   
      random_strg = random.uniform(-1.1, 1.1)
      X_swirl=skswirl(X_scaled,strength=random_strg, preserve_range=True,mode="edge")
      Y_swirl=skswirl(Y_scaled,strength=random_strg, preserve_range=True,mode="edge", order=0) 
      Y_swirl=(Y_swirl>0).astype(tp)
      X_agu=expand_concat(X_agu,X_swirl)
      Y_agu=expand_concat(Y_agu,Y_swirl)
      # Contrast stretching
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
#      Y_rescale=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      ext1 = random.uniform(0,5)
#      ext2 = random.uniform(0,5)
#      p1, p2 = np.percentile(X_scaled, (1+int(ext1), 95+int(ext2)))
#      X_rescale = exposure.rescale_intensity(X_scaled, in_range=(p1, p2))   
#      X_agu=expand_concat(X_agu,X_rescale)
#      Y_agu=expand_concat(Y_agu,Y_rescale) 
#      # Equalization
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
#      Y_eq=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      nbins_rnd = random.uniform(30,1000)
#      X_eq = exposure.equalize_hist(X_scaled,nbins=int(nbins_rnd)) 
#      X_agu=expand_concat(X_agu,X_eq)
#      Y_agu=expand_concat(Y_agu,Y_eq)           
#      # Adajust Log
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge") 
#      Y_adj_log=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      gain_rnd = random.uniform(-5, 5)
#      X_adj_log = exposure.adjust_log(abs(X_scaled),gain=gain_rnd)
#      X_agu=expand_concat(X_agu,X_adj_log)
#      Y_agu=expand_concat(Y_agu,Y_adj_log)        
#      # Adajust Gamma
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")  
#      Y_adj_gamma=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      gain_rnd = random.uniform(-5, 5)
#      gamma_rnd = random.uniform(1, 2)
#      X_adj_gamma = exposure.adjust_gamma(abs(X_scaled),gain=gain_rnd,gamma=int(gamma_rnd))        
#      X_agu=expand_concat(X_agu,X_adj_gamma)
#      Y_agu=expand_concat(Y_agu,Y_adj_gamma)        
#      # Adajust Sigmoid 
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
#      Y_adj_sigmoid=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      gain_rnd = random.uniform(-5, 5)
#      cutoff_rnd = random.uniform(0, 0.5)
#      X_adj_sigmoid=exposure.adjust_sigmoid(abs(X_scaled),gain=gain_rnd,cutoff=cutoff_rnd)
#      X_agu=expand_concat(X_agu,X_adj_sigmoid)
#      Y_agu=expand_concat(Y_agu,Y_adj_sigmoid)   
      # Noise       
      if nosie_flag:
            X_agu=skrandom_noise(X_agu)



      return X_agu,Y_agu
  
def AgumentData_pool(X,Y,nosie,j):
      tp=Y.dtype
      if j==0:
      # Rotate
          X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
          random_degree = random.uniform(-90, 90)
          X_rota=skrotate(X_scaled, random_degree, preserve_range=True,mode="edge")
          Y_rota=skrotate(Y_scaled, random_degree, preserve_range=True,mode="edge")      
          Y_rota=(Y_rota>0).astype(tp)
          X_agu=X_rota
          Y_agu=Y_rota  
#          X_agu=np.expand_dims(np.copy(X_a),axis=0)
#          Y_agu=np.expand_dims(np.copy(Y_a),axis=0)
      elif j==1:

          X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
          random_degree = random.uniform(-45, 45)
          X_rotb=(skrotate(X_scaled.transpose(),random_degree, preserve_range=True,mode="edge")).transpose()
          Y_rotb=(skrotate(Y_scaled.transpose(),random_degree, preserve_range=True,mode="edge")).transpose()
          Y_rotb=(Y_rotb>0).astype(tp)
          X_agu=X_rotb
          Y_agu=Y_rotb  
#          X_agu=expand_concat(X_agu,X_b)
#          Y_agu=expand_concat(Y_agu,Y_b)
      elif j==2:    
          X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
          random_degree = random.uniform(-45, 45)
          X_rotc=skrotate(X_scaled.transpose(0,2,1),random_degree, preserve_range=True,mode="edge").transpose(0,2,1)
          Y_rotc=skrotate(Y_scaled.transpose(0,2,1),random_degree, preserve_range=True,mode="edge").transpose(0,2,1)
          Y_rotc=(Y_rotc>0).astype(tp)
          X_agu=X_rotc
          Y_agu=Y_rotc  
#          X_agu=expand_concat(X_agu,X_c)
#          Y_agu=expand_concat(Y_agu,Y_c)
      elif j==3:
      # Flip
          X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
          X_hflip=X_scaled[:, ::-1,:]
          Y_hflip=Y_scaled[:, ::-1,:]
          Y_hflip=(Y_hflip>0).astype(tp)
          X_agu=X_hflip
          Y_agu=Y_hflip    
#          X_agu=expand_concat(Y_agu,X_hflip)
#          Y_agu=expand_concat(Y_agu,Y_hflip)
      elif j==4:
          X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])
          X_vflip=X_scaled[::-1, :,:]
          Y_vflip=Y_scaled[::-1, :,:]
          Y_vflip=(Y_vflip>0).astype(tp)
          X_agu=X_vflip
          Y_agu=Y_vflip          
#          X_agu=expand_concat(X_agu,X_vflip)
#          Y_agu=expand_concat(Y_agu,Y_vflip)
      elif j==5:
      # Affine
          scale_factor = random.uniform(0.8,1.2)
          random_degree = random.uniform(-0.2, 0.2)
          afftransf=skaffine(shear=random_degree, scale=[scale_factor,scale_factor])
          X_affine=skwarp(X,afftransf, preserve_range=True,mode="edge")
          Y_affine=skwarp(Y,afftransf, preserve_range=True,mode="edge")
          Y_affine=(Y_affine>0).astype(tp)
          X_agu=X_affine
          Y_agu=Y_affine
#          X_agu=expand_concat(X_agu,X_affine)
#          Y_agu=expand_concat(Y_agu,Y_affine)
      elif j==6:
      # Swirl
          X_scaled,Y_scaled=random_dilation(X,Y,factor=[0.8, 1.2])   
          random_strg = random.uniform(-1.1, 1.1)
          X_swirl=skswirl(X_scaled,strength=random_strg, preserve_range=True,mode="edge")
          Y_swirl=skswirl(Y_scaled,strength=random_strg, preserve_range=True,mode="edge") 
          Y_swirl=(Y_swirl>0).astype(tp)
          X_agu=X_swirl
          Y_agu=Y_swirl
#          X_agu=expand_concat(X_agu,X_swirl)
#          Y_agu=expand_concat(Y_agu,Y_swirl)
      # Contrast stretching
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
#      Y_rescale=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      ext1 = random.uniform(0,5)
#      ext2 = random.uniform(0,5)
#      p1, p2 = np.percentile(X_scaled, (1+int(ext1), 95+int(ext2)))
#      X_rescale = exposure.rescale_intensity(X_scaled, in_range=(p1, p2))   
#      X_agu=expand_concat(X_agu,X_rescale)
#      Y_agu=expand_concat(Y_agu,Y_rescale) 
#      # Equalization
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
#      Y_eq=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      nbins_rnd = random.uniform(30,1000)
#      X_eq = exposure.equalize_hist(X_scaled,nbins=int(nbins_rnd)) 
#      X_agu=expand_concat(X_agu,X_eq)
#      Y_agu=expand_concat(Y_agu,Y_eq)           
#      # Adajust Log
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge") 
#      Y_adj_log=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      gain_rnd = random.uniform(-5, 5)
#      X_adj_log = exposure.adjust_log(abs(X_scaled),gain=gain_rnd)
#      X_agu=expand_concat(X_agu,X_adj_log)
#      Y_agu=expand_concat(Y_agu,Y_adj_log)        
#      # Adajust Gamma
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")  
#      Y_adj_gamma=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      gain_rnd = random.uniform(-5, 5)
#      gamma_rnd = random.uniform(1, 2)
#      X_adj_gamma = exposure.adjust_gamma(abs(X_scaled),gain=gain_rnd,gamma=int(gamma_rnd))        
#      X_agu=expand_concat(X_agu,X_adj_gamma)
#      Y_agu=expand_concat(Y_agu,Y_adj_gamma)        
#      # Adajust Sigmoid 
#      scale_factor = random.uniform(0.8, 1.2)
#      afftransf=skaffine(scale=[scale_factor,scale_factor])
#      X_scaled=skwarp(X,afftransf, preserve_range=True,mode="edge")
#      Y_adj_sigmoid=skwarp(Y,afftransf, preserve_range=True,mode="edge")
#      gain_rnd = random.uniform(-5, 5)
#      cutoff_rnd = random.uniform(0, 0.5)
#      X_adj_sigmoid=exposure.adjust_sigmoid(abs(X_scaled),gain=gain_rnd,cutoff=cutoff_rnd)
#      X_agu=expand_concat(X_agu,X_adj_sigmoid)
#      Y_agu=expand_concat(Y_agu,Y_adj_sigmoid)   
      # Noise
      if nosie:
            X_agu=skrandom_noise(X_agu)       


      return X_agu,Y_agu





def agument_data_pool(X_train,Y_train,ncores,noise,parallel_transf,i):

            X_orig_dtype=X_train[i].dtype
            Y_orig_dtype=Y_train[i].dtype
            
            X_reshape = np.reshape(np.squeeze(X_train[i]), list(X_train.shape[1:]) + [1])
            X_reshape = X_reshape.squeeze()
            X_max = np.max(X_reshape)
            Y_reshape = np.reshape(np.squeeze(Y_train[i]), list(Y_train.shape[1:]) + [1])
            Y_reshape = Y_reshape.squeeze()
            if X_orig_dtype == np.float64 or X_orig_dtype == np.float32 or X_orig_dtype == np.float16:
                  pass
            else:
                  print("agument_data_pool: convert to float32")
                  X_reshape=X_reshape.astype(np.float32)
                  Y_reshape=Y_reshape.astype(np.float32)

            if parallel_transf:
    
                func = partial(AgumentData_pool, X_reshape/X_max,Y_reshape, noise)
                pp = Pool(ncores)
                num_transf=7
                MAPP = pp.map(func,range(num_transf))      #for mapped in tqdm(MAP):
                cont=0
                X_ar = np.zeros_like(X_reshape)
                Y_ar = np.zeros_like(Y_reshape)
                for mapped in MAPP:
                      X_ar = mapped[0]
                      Y_ar = mapped[1]
                      cont=cont+1
                      if cont==1:
                                     X_agu=np.expand_dims(np.copy(X_ar),axis=0)
                                     Y_agu=np.expand_dims(np.copy(Y_ar),axis=0)
    
                      X_agu=expand_concat(X_agu,X_ar)         
                      Y_agu=expand_concat(X_agu,Y_ar)   
                close_pool(pp)  
            else:
                            X_agu,Y_agu=AgumentData(X_reshape/X_max,Y_reshape)
                     
            X_out=(np.expand_dims(np.expand_dims(X_agu[0], axis=-1),axis=0)*X_max).astype(X_orig_dtype)
            Y_out=(np.expand_dims(np.expand_dims(Y_agu[0], axis=-1),axis=0)).astype(Y_orig_dtype)
            for j in xrange(1,len(X_agu)):             
                X_aa=np.expand_dims(np.expand_dims(X_agu[j],axis=-1),axis=0)
                Y_aa=np.expand_dims(np.expand_dims(Y_agu[j], axis=-1),axis=0)
                X_aa=(X_aa*X_max).astype(X_orig_dtype)
                Y_aa=Y_aa.astype(Y_orig_dtype)
                X_out=np.concatenate([X_out,X_aa],axis=0)
                Y_out=np.concatenate([Y_out,Y_aa],axis=0)

            return X_out,Y_out





def agument_data(X_train,Y_train,iamge_type_,label_type_,ncores=2,nosie=False,parallel_transf=False):
      
      #print("Data agumentation...")
      X_ag=np.copy(X_train) 
      Y_ag=np.copy(Y_train)
      if parallel_transf:
          for i in xrange(X_train.shape[0]):
              X_out,Y_out=agument_data_pool(X_train, Y_train,ncores,nosie,parallel_transf,i)
              X_ag=np.concatenate([X_ag,X_out],axis=0)
              Y_ag=np.concatenate([Y_ag,Y_out],axis=0)  
          
      else:
              
          func = partial(agument_data_pool, X_train, Y_train,ncores,nosie,parallel_transf)
          p = Pool(ncores)
          MAP = p.map(func,range(X_train.shape[0]))
          
          #for mapped in tqdm(MAP):
          for mapped in MAP:        
              X_a = mapped[0]
              Y_a = mapped[1]
              X_ag=np.concatenate([X_ag,X_a],axis=0)
              Y_ag=np.concatenate([Y_ag,Y_a],axis=0)                       
                      
          
          close_pool(p)            
      return  X_ag.astype(iamge_type_),Y_ag.astype(label_type_)



def deconv3d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides=[1, 2, 2, 2, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    if type(output_size) == int:
        output_size_x=output_size
        output_size_y=output_size
        output_size_z=output_size
    elif type(output_size) == list or type(output_size) == tuple:
        output_size_x=output_size[0]
        output_size_y=output_size[1]
        output_size_z=output_size[2]        
    elif type(output_size) is tf.Dimension:
        if type(output_size.value) == int:
                    output_size_x=output_size
                    output_size_y=output_size
                    output_size_z=output_size
        elif type(output_size.value) == list or type(output_size) == tuple:
                    output_size_x=output_size[0]
                    output_size_y=output_size[1]
                    output_size_z=output_size[2]
    else:
            print("value: "+str(type(output_size.value)))
    out_shape = tf.stack([batch_size, output_size_x, output_size_y, output_size_z, out_channels])
    filter_shape = [filter_size, filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv3d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1



def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides=[1, 2, 2, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    if type(output_size) == int:
        output_size_x=output_size
        output_size_y=output_size
    elif type(output_size) == list or type(output_size) == tuple:
        output_size_x=output_size[0]
        output_size_y=output_size[1]
    elif type(output_size) is tf.Dimension:
            if type(output_size.value) == int:
                    output_size_x=output_size
                    output_size_y=output_size
            elif type(output_size.value) == list or type(output_size) == tuple:
                    output_size_x=output_size[0]
                    output_size_y=output_size[1]
        

    out_shape = tf.stack([batch_size, output_size_x, output_size_y, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1


def conv3d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    return tf.layers.conv3d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding,
                            activation=tf.nn.relu, name=name)



def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding,
                            activation=tf.nn.relu, name=name)
def concatenate(branches):
    return array_ops.concat(branches, 4)
def concatenate2d(branches):
    return array_ops.concat(branches, 3)

def conv3d_block(input_,filters,kernel_size,padding,activation, name_cv1,name_cv2,batch_norm=False,print_shape=False):

            conv1 = tf.layers.conv3d(inputs=input_,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       activation=activation,
                                       name=name_cv1)
            if batch_norm:
				conv1 = tf.layers.batch_normalization(conv1)
            conv2 = tf.layers.conv3d(inputs=conv1,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       activation=activation,
                                       name=name_cv2)
            if batch_norm:
				conv2 = tf.layers.batch_normalization(conv2)
            if print_shape:
				print("shape "+name_cv1+":" +str(conv1.shape))
				print("shape "+name_cv2+":" +str(conv2.shape))
            return conv2 


def full_block(input_,filters,kernel_size,padding,activation, pool_size,name_cv1,name_cv2,name_pool,batch_norm,dropout_layer,print_shape=False,drop_rate=0.5,drop_training=False,name_dropout=None):
              
              
            conv_b=conv3d_block(input_,filters,kernel_size,'SAME',
								tf.nn.relu, name_cv1,name_cv2,
								batch_norm,print_shape)  
            pool = tf.layers.max_pooling3d(inputs=conv_b,
                                              pool_size=pool_size,
                                              strides=2,
                                              name=name_pool)
            if print_shape:
				print("shape "+name_pool+":" +str(pool.shape))
            if dropout_layer:
			    pool = tf.layers.dropout(pool,
                                          drop_rate,
                                          training=drop_training,
                                          name=name_dropout)
			    if print_shape:
					print("shape "+name_dropout+":" +str(dropout_.shape))
            return pool, conv_b 

def deconv3d_block(input_1,input_2,dimm,filt1,filt2,
					filters,kernel_size,padding,activation,
					deconv_name, name_cv1,name_cv2,
					batch_norm,dropout_layer,print_shape=False,
					drop_rate=0.5,drop_training=False,name_dropout=None):
						
            deconv = deconv3d(input_1, 1, dimm, filt1, filt2, deconv_name )  
            deconv1 = concatenate([deconv, input_2])
            
            if dropout_layer:
			    deconv1 = tf.layers.dropout(deconv1,
                                          drop_rate,
                                          training=drop_training,
                                          name=name_dropout)
             
            if print_shape:
				print("shape "+deconv_name+": "+str(deconv.shape))
				print("shape "+deconv_name+" concat: "+str(deconv1.shape))      
            conv_b=conv3d_block(deconv1,filters,kernel_size,padding,
								activation, name_cv1,name_cv2,
								batch_norm,print_shape)  	
            return conv_b	

#%%
#######################################################################################################################
######################################            CLASSES           ###################################################
#######################################################################################################################

      
class ConfusionMatrix(object):
    k = 1
    TPimg = None
    TP = None
    TNimg = None
    TN = None
    FNimg = None
    FN = None
    FPimg = None
    FP = None
    ALLimg = None
    accuracy = None
    Matrix = np.zeros([2, 2])
    
    def __init__(self, result, label, **kwargs):
        k = kwargs.get('k', self.k)
        result = (result == k).astype("int")
        label = (label == k).astype("int")
        sum_img = (result + label)
        self.TPimg = (sum_img == 2).astype("int")
        self.TP = np.count_nonzero(self.TPimg)
        self.TNimg = (sum_img == 0).astype("int")
        self.TN = np.count_nonzero(self.TNimg)
        diff_img = (result - label)
        self.FNimg = (diff_img == -1).astype("int")
        self.FN = np.count_nonzero(self.FNimg)
        self.FPimg = (diff_img == 1).astype("int")
        self.FP = np.count_nonzero(self.FPimg)
        self.ALLimg = self.TPimg + self.TNimg + self.FNimg + self.FPimg
        self.ALL = np.count_nonzero(self.ALLimg)
        
        self.Matrix[0, 0] = self.TP
        self.Matrix[0, 1] = self.FP
        self.Matrix[1, 0] = self.FN
        self.Matrix[1, 1] = self.TN
        
        # Derivations from the Confusion Matrix
        
        # Accuracy (ACC)
        self.accuracy = float(float(float(self.TP) + float(self.TN)) / float(self.ALL))
        # Sensitivity, recall, hit rate, or true positive rate(TPR)
        self.sensitivity = float(float(float(self.TP) )  / (float(self.TP) +float(self.FN)  ) )
        # Specificity, selectivity or true negative rate (TNR)
        self.specificity = float(float(float(self.TN) ) /  (float(self.TN) +float(self.FP)  ) )
        # Precision or positive predictive value (PPV)
        self.precision = float(float(float(self.TP) )  /   (float(self.TP) +float(self.FP)  ) )
        # Negative predictive value (NPV)
        self.NPV = float(float(float(self.TN) ) /  (float(self.TN) +float(self.FN)  ) )
        # Miss rate or false negative rate (FNR)
        self.miss_rate = 1 - self.sensitivity
        # Fall-Out or false positive rate (FPR)
        self.fall_out = 1 - self.specificity
        # False discovery rate (FDR)
        self.FDR =  1 - self.precision
        # False omission rate(FOR)
        self.FOR = 1 - self.NPV
        # F1 score is the harmonic mean of precision and sensitivity
        self.F1_score = float ( ( 2 * self.sensitivity * self.specificity ) / ( self.sensitivity + self.specificity ))
        # Youden J index or Bookmaker Informedness
        self.Youden_index = self.sensitivity + self.specificity -1
        # Matthews correlation coefficient (MCC)
        self.MCC = float (( self.TP * self.TN - self.FP * self.FN) / math.sqrt(float(self.TP+self.FP)*float(self.TP+self.FN)*float(self.TN+self.FP)*float(self.TN+self.FN)))
        # Markedness (MK)
        self.MK = self.precision + self.NPV -1
        # Dice Score 
        self.Dice_score = float( 2 * self.TP) / float( 2 * self.TP + self.FP + self.FN )
    
    def plot(self):
        labels = ["TRUE POSITIVE", "TRUE NEGATIVE", "FALSE NEGATIVE", "FALSE POSITIVE"]
        
        Total = self.TPimg + self.TNimg * 2 + self.FNimg * 3 + self.FPimg * 4
        if len(Total.shape) == 2:
              plt.figure()
              im = plt.imshow(Total, interpolation='none', cmap=plt.get_cmap("Set1"))
              values = np.unique(Total.ravel())
              colors = [im.cmap(im.norm(value)) for value in values]
              # create a patch (proxy artist) for every color
              patches = [matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
              plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
              plt.grid(True)
              plt.show()
        else:
              cmap=plt.get_cmap("Set1")
              axi,sag,cor = OrtoView(Total,colormap=cmap)
              values = np.unique(Total.ravel())
              colors = [cmap(axi.norm(value)) for value in values]
              patches = [matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
              plt.legend(handles=patches, bbox_to_anchor=(0, -7),loc=2)
              #plt.grid(True)
              #plt.title("accuracy: " + str(self.accuracy))
              #plt.show()



class UNET_3D(object):
   

    iterations = 10000000
    batch_size = 10
    batch_count_thr = 50
    ckpt_basename = 'UNET'
    ckpt_dir = None
    drop_rate = tf.constant(0.75)
    drop_training = True
    towritedir = './graphs/UNET'
    logits = None
    ncores=1
    conv_kernel_size=[3,3,3]
    pool_size=[2,2,2]
    gpu_num=str(0)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True

    def __init__(self, **kwargs):
        
        self.gpu_num = kwargs.get('gpu_num', self.gpu_num)
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=self.gpu_num
        print("Use GPU device number "+self.gpu_num)
        with tf.device('/gpu:'+self.gpu_num):
              
	      
	      self.sess = tf.Session(config=self.config)

              self.loss_values = []   
              self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
              self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
              self.ckpt_step = kwargs.get('ckpt_step', None)
              self.ncores = kwargs.get('ncores', 1) 
              
              if self.ckpt_step is not None:
                    if self.ckpt_dir[-1] != "/":
                          self.ckpt_dir=self.ckpt_dir+"/"
                    checkpoint_file=self.ckpt_dir+"checkpoint"
                    if type(self.ckpt_step) is int:
                          self.ckpt_step=str(self.ckpt_step)
                    elif  type(self.ckpt_step) is str:
                          pass
                    else:
                         raise ValueError("ckpt_step must be string or int type") 
                    model_checkpoint_path="model_checkpoint_path: "+"\""+self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+"\""
                    all_model_checkpoint_paths="all_model_checkpoint_paths: "+"\""+self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+"\""
                    
                    
                    if not os.path.exists(self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+".meta"):
                          raise IOError(self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+" does not exists")
                    
                    
                    outF = open(checkpoint_file, "w")
                    for line in [model_checkpoint_path,all_model_checkpoint_paths]:
                          print >>outF, line
                    outF.close()
              self.towritedir = kwargs.get('towritedir', self.towritedir)        
              self.filter1stnumb = kwargs.get('filter1stnumb', 32)
              self.loss_type = kwargs.get('loss_type', "sigmoid_cross_entropy")
              self.dp_rate = kwargs.get('drop_rate', 0.75)
              self.drop_rate = tf.constant(self.dp_rate)
              self.drop_training = kwargs.get('drop_training', True)
              self.gstep = tf.Variable(0, dtype=tf.int32,
                                  trainable=False, name='global_step')
              self.use_softmax = kwargs.get('use_softmax', False)
              self.lr = tf.placeholder(tf.float32)
              self.learning_rate = 0.0001
              self.accuracy = 0.0
              self.dice = 0.0
    
              self.conv_kernel_size = kwargs.get('conv_kernel_size', self.conv_kernel_size) 
    
    def summary(self):
        '''
        Create summaries to write on TensorBoard
        track both training loss and test accuracy
        '''
        with tf.device('/gpu:'+self.gpu_num):
              with tf.name_scope('summaries'):
                      tf.summary.scalar('loss', self.loss)
                      tf.summary.scalar('accuracy', self.accuracy)
                      tf.summary.histogram('histogram_loss', self.loss)
                      self.summary_op = tf.summary.merge_all()
    
    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]
              conv1 = tf.layers.conv3d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv3d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling3d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv3d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv3d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling3d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv3d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv3d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling3d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv3d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv3d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling3d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv3d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv3d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape))              
              dropout = tf.layers.dropout(conv10,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))
              deconv1 = deconv3d(dropout, 1, self.IMG_WIDTH / 8, self.filter1stnumb*8, self.filter1stnumb*16, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate([deconv1, conv8]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv11 = tf.layers.conv3d(inputs=deconv1,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              
              conv12 = tf.layers.conv3d(inputs=conv11,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv12')             
              print("shape conv12: "+str(conv12.shape))              
              deconv2 = deconv3d(conv12, 1,self.IMG_WIDTH / 4, self.filter1stnumb*4, self.filter1stnumb*8, "deconv2")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate([deconv2, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv13 = tf.layers.conv3d(inputs=deconv2,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv3d(inputs=conv13,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv14')
              print("shape conv14: "+str(conv14.shape))              
              deconv3 = deconv3d(conv14, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv3: "+str(deconv3.shape))
              deconv3 = concatenate([deconv3, conv4]) 
              print("shape deconv3 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv3d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              conv16 = tf.layers.conv3d(inputs=conv15,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv16')
              print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv3d(conv16, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate([deconv4, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv3d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              conv18 = tf.layers.conv3d(inputs=conv17,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv18')
              print("shape conv18: "+str(conv18.shape))       
              logits = tf.layers.conv3d(inputs=conv18,
                                        filters=1,
                                        kernel_size=[1, 1, 1],
                                        padding='SAME',
                                        activation=tf.nn.sigmoid,
                                        name='logits')
              print("shape logits: "+str(logits.shape))       
              return logits
    


    def pixel_wise_softmax(self,output_map):
        with tf.name_scope("pixel_wise_softmax"):
            max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
            exponential_map = tf.exp(output_map - max_axis)
            normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
            return exponential_map / normalize
    
    
    def shuffle(self):
        p = np.random.permutation(len(self.X_train))
        self.images = self.X_train[p]
        self.labels = self.Y_train[p]
    
    def next_batch(self, iters):
        if (iters == 0):
            self.shuffle()
        count = self.batch_size * iters
        return self.images[count:(count + self.batch_size)], self.labels[count:(count + self.batch_size)]
    
    def agument_batch(self,X,Y,parallel_transf=False):
        return agument_data(X,Y,self.image_type,self.label_type,ncores=self.ncores,parallel_transf=parallel_transf )
    
    def initialize_XY(self,X_t,Y_t=None):

                  self.IMG_WIDTH = X_t[0].shape[0]
                  self.IMG_HEIGHT = X_t[0].shape[1]
                  self.IMG_LENGTH = X_t[0].shape[2]

		  
                  Y_type=tf.float32
		  X_type=tf.float32
		  if X_t.dtype == np.float16:
			X_type=tf.float16
		  if Y_t is not None:
		  	if Y_t.dtype == np.float16:
				Y_type=tf.float16

		  if  self.IMG_LENGTH != 1: 
                  	
                  	self.X = tf.placeholder(X_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
                  	if Y_t is not None:
                          self.Y_ = tf.placeholder(Y_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
                          self.labels = Y_t
                          self.label_type=Y_t.dtype
                  	self.dims=[self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH]
		  else:
                  	self.X = tf.placeholder(X_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
                  	if Y_t is not None:
                          self.Y_ = tf.placeholder(Y_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
                          self.labels = Y_t
                          self.label_type=Y_t.dtype
                  	self.dims=[self.IMG_WIDTH, self.IMG_HEIGHT]
                  self.images  = X_t 
                  self.image_type = X_t.dtype
		          

    def train(self,X_train,Y_train, **kwargs):
        
        with tf.device('/gpu:'+self.gpu_num):  
            
            
              self.X_train = X_train
              self.Y_train = Y_train

              if self.logits is None:
                      self.initialize_XY(X_train,Y_train)
            
            
            
              self.iterations = kwargs.get('iter', self.iterations)
              self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
              self.batch_size = kwargs.get('batch_size', self.batch_size)
              self.batch_count_thr = kwargs.get('batch_count_thr', int(self.X_train.shape[0]/self.batch_size))
              self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
              self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
              self.skip_iter = kwargs.get('skip_iter', int(float(self.iterations*0.1)))      
              self.agument_batch_flag = kwargs.get('agument_batch', False) 
              self.parallel_transf = kwargs.get('parallel_transf', False) 
              if self.skip_iter == 0:
                    self.skip_iter=1
              if self.batch_count_thr == 0:
                    self.self.batch_count_thr=1        
              print("iterations: " + str(self.iterations))
              print("skip every "+str(self.skip_iter)+" iterations" )
              print("batch size: "+str(self.batch_size) )
              print("batch threshold: "+str(self.batch_count_thr) )
              
              if self.logits is None:
                    self.logits = self.build_model(self.X)
                    if self.use_softmax:
                          predicted=self.pixel_wise_softmax(self.logits)
                    else:
                          predicted=self.logits
                    if self.loss_type == "sigmoid_cross_entropy":
                          self.loss = tf.losses.sigmoid_cross_entropy(self.Y_,predicted)  
                    
                    elif self.loss_type == "softmax_cross_entropy":
                           self.loss = tf.losses.softmax_cross_entropy(tf.cast(self.Y_, tf.int32),predicted)

		    elif self.loss_type == "sparse_softmax_cross_entropy":
			   self.loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.Y_, tf.int32),predicted ) 
                                       
 		    elif self.loss_type == "softmax_cross_entropy_with_logits":
			   self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y_,logits=predicted )                         
                    
                    elif self.loss_type == "dice_score": 
                          self.loss = 1 - dice_coe(predicted,self.Y_,smooth=1e-8)
                    
                    elif self.loss_type == "mean_squared_error":                          
                          self.loss = tf.losses.mean_squared_error(self.Y_,predicted)
                    
                    elif self.loss_type == "focal_loss":
                          #self.loss = focal_loss(self.logits,tf.cast(self.Y_, tf.float32), gamma=2)

#                          self.loss = focal_loss_on_object_detection(self.Y_,predicted , gamma=2)
                          self.loss = focal_loss_on_object_detection(self.Y_,predicted , gamma=2)
                    
                    elif self.loss_type == "cosine_distance":
                          self.loss = tf.losses.cosine_distance(self.Y_,predicted,weights=1.0,axis=1)
                    
                    elif self.loss_type == "absolute_difference":
                          self.loss = tf.losses.absolute_difference(self.Y_,predicted,weights=1.0)
                    else:
                          raise NameError('Invalid Loss Function')
                    
                    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
              
              # Summary
              self.summary()
              
              # init
              init = tf.global_variables_initializer()
              
              self.sess.run(init)
              saver = tf.train.Saver()
              if self.ckpt_dir:
                  trymakedir(self.ckpt_dir)
                  ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
                  
                  if ckpt and ckpt.model_checkpoint_path:
                      saver.restore(self.sess, ckpt.model_checkpoint_path)
              try:
                  step = self.gstep.eval(session=self.sess)
              except:
                  step = 0
                  pass
              
              batch_count = 0
              writer = tf.summary.FileWriter(self.towritedir, tf.get_default_graph())
              
              for i in xrange(self.iterations):
                  # training on batches 
                  if (batch_count > self.batch_count_thr):
                      batch_count = 0
                  batch_X, batch_Y = self.next_batch(batch_count)
                  batch_X = StandData(batch_X, batch_X)
                  if self.agument_batch_flag:
                       batch_X_ag, batch_Y_ag = self.agument_batch(batch_X, batch_Y,parallel_transf=self.parallel_transf)
                       batch_X = StandData(batch_X_ag, batch_X)
                       batch_Y = batch_Y_ag
                  batch_X=NormImages(batch_X)
                  batch_count += 1
                  feed_dict = {self.X: batch_X, self.Y_: batch_Y, self.lr: self.learning_rate}
                  loss_value, _,summaries = self.sess.run([self.loss, self.optimizer, self.summary_op], feed_dict=feed_dict)
                  writer.add_summary(summaries, global_step=step)
                  self.loss_values.append(loss_value)
                  if (i % self.skip_iter == 0):
                      print(str(step) + " training loss:", str(loss_value))
                      step += 1
                      if self.ckpt_dir:

                          saver.save(self.sess, self.ckpt_dir + self.ckpt_basename, step)
              writer.close()
              print("Done!")
    
    def test(self, X_test, Y_test,**kwargs):
        


        if self.logits is None:

            self.initialize_XY( X_test, Y_test)

            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
        
        plot_flag = kwargs.get('plot', False)
        metrics_flag = kwargs.get('compute_metrics', True)
        acc_flag = kwargs.get('compute_accuracy',True)
        plot_accuracy=kwargs.get('plot_accuracy',True)
        if metrics_flag:
              acc_flag=True
        i=-1
        images_dim=[len(indices)]+self.dims
        test_image=np.zeros(images_dim)
        result_mask_seqeeze_orig=np.zeros(images_dim)
        dice_v=np.zeros([len(indices)])                       
        accuracy_v=np.zeros([len(indices)]) 
        if metrics_flag:
              MCC_v=np.zeros([len(indices)])
              Youden_index_v=np.zeros([len(indices)])
              sensitivity_v=np.zeros([len(indices)])
              specificity_v=np.zeros([len(indices)])
              pearscoeff_v=np.zeros([len(indices)])
              f1_score_v=np.zeros([len(indices)])
              precision_v=np.zeros([len(indices)])
              dice_score_v=np.zeros([len(indices)])
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = np.squeeze(test_image_i)
              test_mask_i = Y_test[ix].astype(np.float32)
              test_mask = np.squeeze(test_mask_i.astype(np.int16))
              rsp_dim= [-1] +self.dims +[ 1]
              rslt_dim = self.dims + [1]
              test_image_reshape = np.reshape(test_image[i],rsp_dim)
              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([self.logits], feed_dict=test_data)
              result_mask = np.reshape(np.squeeze(result_mask), rslt_dim)
              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])
              
              
              if acc_flag or plot_accuracy:
                    
                    ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                    accuracy_v[i]=ConfMat.accuracy 
              if metrics_flag:
                    
                    
                    MCC_v[i]=ConfMat.MCC
                    Youden_index_v[i] = ConfMat.Youden_index
                    sensitivity_v[i]  = ConfMat.sensitivity
                    specificity_v[i]  = ConfMat.specificity
                    f1_score_v[i]     = ConfMat.F1_score
                    precision_v[i]    = ConfMat.precision
                    rpea,ppea = scipy.stats.pearsonr(result_mask_seqeeze_int.flatten(), test_mask.flatten())
                    pearscoeff_v[i]   = rpea
                    self.dice = np.mean(dice_v) 
                    dice_score_v[i]     = ConfMat.Dice_score
              if plot_accuracy:
                          ConfMat.plot(),plt.suptitle("index: "+str(ix)+" - Accuracy: "+'%.3f' % ConfMat.accuracy)
                          if metrics_flag:
                                      plt.suptitle("index: "+str(ix)+" - Accuracy: "+'%.3f' % ConfMat.accuracy+" - Dice Score: "+'%.3f' % ConfMat.Dice_score)
        self.accuracy =  np.mean(accuracy_v)            
        
        
        
        if metrics_flag:          
              self.MCC =  np.mean(MCC_v)
              self.Youden_index =  np.mean(Youden_index_v)
              self.sensitivity  =  np.mean(sensitivity_v)
              self.specificity  =  np.mean(specificity_v)
              self.pearscoeff   =  np.mean(pearscoeff_v)
              self.f1_score     =  np.mean(f1_score_v)
              self.precision    =  np.mean(precision_v)
              self.dice_score   =  np.mean(dice_score_v)
              print 'Sensitivity (recall):    {}'.format(round(self.sensitivity,4))
              print 'Specificity:             {}'.format(round(self.specificity,4))
              print 'Accuracy:                {}'.format(round(self.accuracy,4))               
              print 'Dice Score:              {}'.format(round(self.dice_score,4))
              print 'Precision:               {}'.format(round(self.precision,4))               
              print 'F1 score:                {}'.format(round(self.f1_score,4))
              print 'MCC:                     {}'.format(round(self.MCC,4))
              print 'Youden index:            {}'.format(round(self.Youden_index,4)) 
              print 'Pearson coef:            {}'.format(round(self.pearscoeff,4))
        
        if plot_flag:
            i=-1
            for ix in indices: 
                  
                  i=i+1
                  test_mask_i = Y_test[ix].astype(float)
                  test_mask = np.squeeze(test_mask_i)
                  
                  
                  result_mask_seqeeze =result_mask_seqeeze_orig[i].astype(np.float32)
                  
                  result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])
                  if len(self.dims) == 3:
                      OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("Result Mask (orig)")
                      OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("result mask int")
                      OrtoView(test_image[i], Mask=result_mask_seqeeze_orig[i]), plt.suptitle("Test image vs  Result Mask (orig - overlay)")
                      OrtoView(test_image[i], Mask=result_mask_seqeeze), plt.suptitle("Test image vs  Result Mask (overlay)")   
                      OrtoView(test_mask, Mask=result_mask_seqeeze,colormap=plt.get_cmap("Greens"))
                      OrtoView(test_image[i],Mask=test_mask), plt.suptitle("Test image vs  Test Mask (GT)")
                      OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("Test image vs  Result Mask int")
                      OrtoView(test_mask, colormap=plt.get_cmap("Greens"),Mask=result_mask_seqeeze_int), plt.suptitle("Test Mask vs  Result Mask int")
                  else:
                      plt.figure(),plt.imshow(result_mask_seqeeze_orig[i], cmap=matplotlib.cm.jet), plt.suptitle("Result Mask (orig)")
                      plt.figure(),plt.imshow(test_image[i], cmap=plt.get_cmap("gray")), plot_overlay(result_mask_seqeeze_orig[i],alpha=0.5), plt.suptitle("Test image vs  Result Mask (overlay)") 
                      plt.figure(),plt.imshow(test_image[i], cmap=plt.get_cmap("gray")), plot_overlay(result_mask_seqeeze_int,alpha=0.5), plt.suptitle("Test image vs  Result Mask int (overlay)") 
        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig     


    def predict(self, X_input, **kwargs):
        
        if self.logits is None:
            
            self.initialize_XY( X_input)
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_input.shape[0]))
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
        
        plot_flag = kwargs.get('plot', False)

        i=-1
        rsp_dim= [-1] +self.dims +[ 1]
        rslt_dim = self.dims + [1]
        test_image_reshape = np.reshape(X_input[i],rsp_dim)
        images_dim=[len(indices)]+self.dims
        result_mask_seqeeze_orig=np.zeros(images_dim)    
        test_image=np.zeros(images_dim)
        for ix in indices:
              i=i+1
              test_image_i = X_input[ix].astype(np.float32)
              test_image[i] = np.squeeze(test_image_i)             
              test_image_reshape = np.reshape(test_image[i], rsp_dim)              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([self.logits], feed_dict=test_data)
              result_mask = np.reshape(np.squeeze(result_mask), rslt_dim)              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])
               
        

        if plot_flag:
            
            for ix in indices:        
                                  
                  i=i+1
                  result_mask_seqeeze =result_mask_seqeeze_orig[i].astype(np.float32)     
                  result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])                  
                  OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("Result Mask (orig)")
                  OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("result mask int")
                  OrtoView(test_image[i], Mask=result_mask_seqeeze_orig[i]), plt.suptitle("Test image vs  Result Mask (orig - overlay)")
                  OrtoView(test_image[i], Mask=result_mask_seqeeze), plt.suptitle("Test image vs  Result Mask (overlay)")   
                  OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("Test image vs  Result Mask int")

        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig         








    
class UNET_3D_seg(UNET_3D):
    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]
              conv1 = tf.layers.conv3d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv3d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling3d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv3d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv3d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling3d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv3d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv3d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling3d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv3d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv3d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling3d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv3d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv3d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape))              
              dropout = tf.layers.dropout(conv10,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))
              deconv1 = deconv3d(dropout, 1, self.IMG_WIDTH / 8, self.filter1stnumb*8, self.filter1stnumb*16, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate([deconv1, conv8]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv11 = tf.layers.conv3d(inputs=deconv1,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              
              conv12 = tf.layers.conv3d(inputs=conv11,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv12')             
              print("shape conv12: "+str(conv12.shape))              
              deconv2 = deconv3d(conv12, 1,self.IMG_WIDTH / 4, self.filter1stnumb*4, self.filter1stnumb*8, "deconv2")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate([deconv2, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv13 = tf.layers.conv3d(inputs=deconv2,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv3d(inputs=conv13,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv14')
              print("shape conv14: "+str(conv14.shape))              
              deconv3 = deconv3d(conv14, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv3: "+str(deconv3.shape))
              deconv3 = concatenate([deconv3, conv4]) 
              print("shape deconv3 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv3d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              conv16 = tf.layers.conv3d(inputs=conv15,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv16')
              print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv3d(conv16, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate([deconv4, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv3d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              conv18 = tf.layers.conv3d(inputs=conv17,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv18')
              print("shape conv18: "+str(conv18.shape))       
              logits = tf.layers.conv3d(inputs=conv18,
                                        filters=1,
                                        kernel_size=[1, 1, 1],
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='logits')
              print("shape logits: "+str(logits.shape))       
              return logits
          
    def test(self, X_test, Y_test,**kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test[0].shape[0]
            self.IMG_HEIGHT = X_test[0].shape[1]
            self.IMG_LENGTH = X_test[0].shape[2]
            self.images, self.labels = X_test, Y_test
            self.image_type = X_test.dtype
            self.label_type=Y_test.dtype
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
              
        seg_labels=kwargs.get('labels',None)
       

        
        plot_flag = kwargs.get('plot', False)
        metrics_flag = kwargs.get('compute_metrics', True)
        acc_flag = kwargs.get('compute_accuracy',True)
        plot_accuracy=kwargs.get('plot_accuracy',True)
        plot_separately_flag=kwargs.get('plot_separately',False)
        if metrics_flag:
              acc_flag=True
        i=-1
        test_image=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])
        result_mask_seqeeze_orig=np.zeros([len(indices),Y_test[0].shape[0],Y_test[0].shape[1],Y_test[0].shape[2]])
        dice_v=np.zeros([len(indices)])                       
        accuracy_v=np.zeros([len(indices)]) 
        if metrics_flag:
              MCC_v=np.zeros([len(indices)])
              Youden_index_v=np.zeros([len(indices)])
              sensitivity_v=np.zeros([len(indices)])
              specificity_v=np.zeros([len(indices)])
              pearscoeff_v=np.zeros([len(indices)])
              f1_score_v=np.zeros([len(indices)])
              precision_v=np.zeros([len(indices)])
              dice_score_v=np.zeros([len(indices)])
              
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = test_image_i[:, :, :, 0]
              test_mask_i = Y_test[ix].astype(np.float32)
              test_mask_all = test_mask_i[:, :, :, 0].astype(np.int16)
             
              if seg_labels is None:
                  seg_labels=np.unique(test_mask_all)
                  
                  if np.any(seg_labels==0.0):
                      seg_labels=np.delete(seg_labels, 0)
                  print("labels found: "+str(seg_labels))
                  
                  
              if isinstance(seg_labels, list) :
                      seg_labels.sort()
              elif isinstance(seg_labels, np.ndarray):
                      np.sort(seg_labels)
              else:
                      seg_labels=[seg_labels]

              test_image_reshape = np.reshape(test_image[i], [-1, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([self.logits], feed_dict=test_data)
              result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i])
              accuracy_vv=np.zeros([len(seg_labels)])
              MCC_vv=np.zeros([len(seg_labels)])
              Youden_index_vv=np.zeros([len(seg_labels)])
              sensitivity_vv=np.zeros([len(seg_labels)])
              specificity_vv=np.zeros([len(seg_labels)])
              f1_score_vv=np.zeros([len(seg_labels)])
              precision_vv=np.zeros([len(seg_labels)])
              pearscoeff_vv=np.zeros([len(seg_labels)])
              dice_score_vv=np.zeros([len(seg_labels)])
              if acc_flag :
                       for idx,value in enumerate(seg_labels): 
                           result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                           test_mask=(test_mask_all==value).astype(np.int)
                           ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                           accuracy_vv[idx]=ConfMat.accuracy
                       accuracy_v[i]=np.mean(accuracy_vv)
              if metrics_flag:
                                for idx,value in enumerate(seg_labels):
                                    result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                                    test_mask=(test_mask_all==value).astype(np.int)
                                    ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)

                                    MCC_vv[idx]=ConfMat.MCC
                                    Youden_index_vv[idx] = ConfMat.Youden_index
                                    sensitivity_vv[idx]  = ConfMat.sensitivity
                                    specificity_vv[idx]  = ConfMat.specificity
                                    f1_score_vv[idx]     = ConfMat.F1_score
                                    precision_vv[idx]    = ConfMat.precision
                                    rpea,ppea = scipy.stats.pearsonr(result_mask_seqeeze_int.flatten(), test_mask.flatten())
                                    pearscoeff_vv[idx]   = rpea
                                    dice_score_vv[idx]     = ConfMat.Dice_score
                        
                                MCC_v[i]          = np.mean(MCC_vv)
                                Youden_index_v[i] = np.mean(Youden_index_vv)
                                sensitivity_v[i]  = np.mean(sensitivity_vv)
                                specificity_v[i]  = np.mean(specificity_vv)
                                f1_score_v[i]     = np.mean(f1_score_vv)
                                precision_v[i]    = np.mean(precision_vv)                                
                                pearscoeff_v[i]   = np.mean(pearscoeff_vv)
                                dice_score_v[i]   = np.mean(dice_score_vv)
                    
              if plot_accuracy:
                           for idx,value in enumerate(seg_labels):
                               result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                               test_mask=(test_mask_all==value).astype(np.int)
                               ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                               ConfMat.plot(),plt.suptitle("index: "+str(ix)+" -  label:"+str(value)+" - Accuracy: "+'%.3f' % ConfMat.accuracy + "- Dice Score: "+'%.3f' % ConfMat.Dice_score)
        
        self.accuracy =  np.mean(accuracy_v)            
        
        
        
        if metrics_flag:          
              self.MCC =  np.mean(MCC_v)
              self.Youden_index =  np.mean(Youden_index_v)
              self.sensitivity  =  np.mean(sensitivity_v)
              self.specificity  =  np.mean(specificity_v)
              self.pearscoeff   =  np.mean(pearscoeff_v)
              self.f1_score     =  np.mean(f1_score_v)
              self.precision    =  np.mean(precision_v)
              self.dice_score   =  np.mean(dice_score_v)
              print 'Sensitivity (recall):    {}'.format(round(self.sensitivity,4))
              print 'Specificity:             {}'.format(round(self.specificity,4))
              print 'Accuracy:                {}'.format(round(self.accuracy,4))               
              print 'Dice Score:              {}'.format(round(self.dice_score,4))
              print 'Precision:               {}'.format(round(self.precision,4))               
              print 'F1 score:                {}'.format(round(self.f1_score,4))
              print 'MCC:                     {}'.format(round(self.MCC,4))
              print 'Youden index:            {}'.format(round(self.Youden_index,4)) 
              print 'Pearson coef:            {}'.format(round(self.pearscoeff,4))
        
        if plot_flag:
            
            for ix in indices: 
                 test_mask_i = Y_test[ix].astype(float)
                 test_mask = test_mask_i[:, :, :, 0]
                 OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Result Segmentation (orig)")
                 OrtoView(np.round(result_mask_seqeeze_orig[i]), colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Result Segmentation (round)")
                 
                 if plot_separately_flag:
                     for idx,value in enumerate(seg_labels):
                          result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                          test_mask=(test_mask_all==value).astype(np.int)                
                      
                         # result_mask_seqeeze =result_mask_seqeeze_orig[i].astype(np.float32)
                          
      
                          
                          OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("label: "+str(value)+" result mask int")
                          #OrtoView(test_image[i], Mask=result_mask_seqeeze_orig[i]), plt.suptitle("Test image vs  Result Mask (orig - overlay)")
                          #OrtoView(test_image[i], Mask=result_mask_seqeeze), plt.suptitle("Test image vs  Result Mask (overlay)")   
                          #OrtoView(test_mask, Mask=result_mask_seqeeze,colormap=plt.get_cmap("Greens"))
                          OrtoView(test_image[i],Mask=test_mask), plt.suptitle("label: "+str(value)+" Test image vs  Test Mask (GT)")
                          OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Test image vs  Result Mask int")
                          OrtoView(test_mask, colormap=plt.get_cmap("Greens"),Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Test Mask vs  Result Mask int")
            if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig

    def predict(self, X_test, **kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test[0].shape[0]
            self.IMG_HEIGHT = X_test[0].shape[1]
            self.IMG_LENGTH = X_test[0].shape[2]
            self.images = X_test
            self.image_type = X_test.dtype          
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
              
        seg_labels=kwargs.get('labels',None)       
        plot_flag = kwargs.get('plot', False)
        plot_separately_flag=kwargs.get('plot_separately',False)
        
        test_image=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])
        result_mask_seqeeze_orig=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])

        i=-1    
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = test_image_i[:, :, :, 0]
  
              test_image_reshape = np.reshape(test_image[i], [-1, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([self.logits], feed_dict=test_data)
              result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i])            
              
              if seg_labels is None:
                  seg_labels=np.unique(result_mask_seqeeze_int_all)
                  
                  if np.any(seg_labels==0.0):
                      seg_labels=np.delete(seg_labels, 0)
                  print("labels found: "+str(seg_labels))
                  
                  
              if isinstance(seg_labels, list) :
                      seg_labels.sort()
              elif isinstance(seg_labels, np.ndarray):
                      np.sort(seg_labels)
              else:
                      seg_labels=[seg_labels]

              

        if plot_flag:
            
            for ix in indices: 
		 i=i+1
                 OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Result Segmentation (orig)")
                 OrtoView(np.round(result_mask_seqeeze_orig[i]), colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Result Segmentation (round)")
                 
                 if plot_separately_flag:
                     for idx,value in enumerate(seg_labels):
                          result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                          OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("label: "+str(value)+" result mask int")
                          OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Test image vs  Result Mask int")
        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig     



class UNET_3D_multiclass(UNET_3D_seg):
    num_labels = 1

    def __init__(self, **kwargs):
        
        self.gpu_num = kwargs.get('gpu_num', self.gpu_num)
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=self.gpu_num
        print("Use GPU device number "+self.gpu_num)
        with tf.device('/gpu:'+self.gpu_num):
              
	      
	      self.sess = tf.Session(config=self.config)

              self.loss_values = []   
              self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
              self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
              self.ckpt_step = kwargs.get('ckpt_step', None)
              self.ncores = kwargs.get('ncores', 1) 
              
              if self.ckpt_step is not None:
                    if self.ckpt_dir[-1] != "/":
                          self.ckpt_dir=self.ckpt_dir+"/"
                    checkpoint_file=self.ckpt_dir+"checkpoint"
                    if type(self.ckpt_step) is int:
                          self.ckpt_step=str(self.ckpt_step)
                    elif  type(self.ckpt_step) is str:
                          pass
                    else:
                         raise ValueError("ckpt_step must be string or int type") 
                    model_checkpoint_path="model_checkpoint_path: "+"\""+self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+"\""
                    all_model_checkpoint_paths="all_model_checkpoint_paths: "+"\""+self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+"\""
                    
                    
                    if not os.path.exists(self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+".meta"):
                          raise IOError(self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+" does not exists")
                    
                    
                    outF = open(checkpoint_file, "w")
                    for line in [model_checkpoint_path,all_model_checkpoint_paths]:
                          print >>outF, line
                    outF.close()
              self.towritedir = kwargs.get('towritedir', self.towritedir)        
              self.filter1stnumb = kwargs.get('filter1stnumb', 32)
              self.loss_type = kwargs.get('loss_type', "sparse_softmax_cross_entropy")
              self.dp_rate = kwargs.get('drop_rate', 0.75)
              self.drop_rate = tf.constant(self.dp_rate)
              self.drop_training = kwargs.get('drop_training', True)
              self.gstep = tf.Variable(0, dtype=tf.int32,
                                  trainable=False, name='global_step')
              self.use_softmax = kwargs.get('use_softmax', False)
              self.lr = tf.placeholder(tf.float32)
              self.learning_rate = 0.0001
              self.accuracy = 0.0
              self.dice = 0.0
              self.dice_score=0.0
              self.sensitivity=0.0
              self.specificity=0.0
    	      self.num_labels = kwargs.get('num_labels', self.num_labels) 	
              self.conv_kernel_size = kwargs.get('conv_kernel_size', self.conv_kernel_size) 

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        track both training loss and test accuracy
        '''
        with tf.device('/gpu:'+self.gpu_num):
              with tf.name_scope('summaries'):
                      tf.summary.scalar('loss', self.loss)
                      tf.summary.scalar('accuracy', np.mean(self.accuracy))
                      tf.summary.scalar('dice_score', np.mean(self.dice_score))
                      tf.summary.scalar('sensitivity', np.mean(self.sensitivity))
                      tf.summary.scalar('specificity', np.mean(self.specificity))
                      tf.summary.histogram('histogram_loss', self.loss)
                      self.summary_op = tf.summary.merge_all()  

    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]
              conv1 = tf.layers.conv3d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv3d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling3d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv3d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv3d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling3d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv3d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv3d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling3d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv3d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv3d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling3d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv3d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv3d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape))              
              dropout = tf.layers.dropout(conv10,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))
              deconv1 = deconv3d(dropout, 1, [self.IMG_WIDTH / 8,self.IMG_HEIGHT / 8  ,self.IMG_LENGTH / 8 ], self.filter1stnumb*8, self.filter1stnumb*16, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate([deconv1, conv8]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv11 = tf.layers.conv3d(inputs=deconv1,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              
              conv12 = tf.layers.conv3d(inputs=conv11,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv12')             
              print("shape conv12: "+str(conv12.shape))              
              deconv2 = deconv3d(conv12, 1, [self.IMG_WIDTH / 4,self.IMG_HEIGHT / 4 ,self.IMG_LENGTH / 4 ] , self.filter1stnumb*4, self.filter1stnumb*8, "deconv2")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate([deconv2, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv13 = tf.layers.conv3d(inputs=deconv2,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv3d(inputs=conv13,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv14')
              print("shape conv14: "+str(conv14.shape))              
              deconv3 = deconv3d(conv14, 1, [self.IMG_WIDTH / 2,self.IMG_HEIGHT / 2 ,self.IMG_LENGTH / 2 ], self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv3: "+str(deconv3.shape))
              deconv3 = concatenate([deconv3, conv4]) 
              print("shape deconv3 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv3d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              conv16 = tf.layers.conv3d(inputs=conv15,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv16')
              print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv3d(conv16, 1, [self.IMG_WIDTH ,self.IMG_HEIGHT ,self.IMG_LENGTH], self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate([deconv4, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv3d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              conv18 = tf.layers.conv3d(inputs=conv17,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv18')
              print("shape conv18: "+str(conv18.shape))       
	      logits = tf.layers.conv3d(inputs=conv18,
                                        filters=self.num_labels,
                                        kernel_size=[1, 1, 1],
                                        padding='SAME',
                                        activation=tf.identity,
                                        name='logits')
              print("shape logits: "+str(logits.shape))       
              return logits

    def test(self, X_test, Y_test,**kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test[0].shape[0]
            self.IMG_HEIGHT = X_test[0].shape[1]
            self.IMG_LENGTH = X_test[0].shape[2]
            self.images, self.labels = X_test, Y_test
            self.image_type = X_test.dtype
            self.label_type=Y_test.dtype
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
              
        seg_labels=kwargs.get('labels',None)
       

        if seg_labels is None:
                       
              test_mask_i = Y_test[0].astype(np.float32)
              test_mask_all = test_mask_i[:, :, :, 0].astype(np.int16)
              
              seg_labels=np.unique(test_mask_all)
                  
              if np.any(seg_labels==0.0):
                      seg_labels=np.delete(seg_labels, 0)
              print("labels found: "+str(seg_labels))
        else:                     
              if isinstance(seg_labels, list) :
                      seg_labels.sort()
              elif isinstance(seg_labels, np.ndarray):
                      np.sort(seg_labels)
              elif isinstance(seg_labels, int):               
                       seg_labels=[seg_labels]                 
              else:
                       seg_labels=[seg_labels]
       
        
        plot_flag = kwargs.get('plot', False)
        metrics_flag = kwargs.get('compute_metrics', True)
        print_metrics_flag = kwargs.get('print_metric', True)
        acc_flag = kwargs.get('compute_accuracy',True)
        plot_accuracy=kwargs.get('plot_accuracy',True)
        plot_separately_flag=kwargs.get('plot_separately',False)
        print_separately_flag=kwargs.get('print_separately',False)
        plot_ground_truth=kwargs.get('plot_ground_truth',False)
        if metrics_flag:
              acc_flag=True
        i=-1
        test_image=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])
        result_mask_seqeeze_orig=np.zeros([len(indices),Y_test[0].shape[0],Y_test[0].shape[1],Y_test[0].shape[2]])
        dice_v=np.zeros([len(indices),len(seg_labels)])                       
        accuracy_v=np.zeros([len(indices),len(seg_labels)]) 
        if metrics_flag:
              MCC_v=np.zeros([len(indices),len(seg_labels)])
              Youden_index_v=np.zeros([len(indices),len(seg_labels)])
              sensitivity_v=np.zeros([len(indices),len(seg_labels)])
              specificity_v=np.zeros([len(indices),len(seg_labels)])
              pearscoeff_v=np.zeros([len(indices),len(seg_labels)])
              f1_score_v=np.zeros([len(indices),len(seg_labels)])
              precision_v=np.zeros([len(indices),len(seg_labels)])
              dice_score_v=np.zeros([len(indices),len(seg_labels)])
              
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = test_image_i[:, :, :, 0]
              test_mask_i = Y_test[ix].astype(np.float32)
              test_mask_all = test_mask_i[:, :, :, 0].astype(np.int16)
             
              if seg_labels is None:
                  seg_labels=np.unique(test_mask_all)
                  
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

              test_image_reshape = np.reshape(test_image[i], [-1, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              test_data = {self.X: test_image_reshape}
              #result_mask = self.sess.run([self.logits], feed_dict=test_data)
	      result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)
      
	      #return result_mask

              result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i])
              accuracy_vv=np.zeros([len(seg_labels)])
              MCC_vv=np.zeros([len(seg_labels)])
              Youden_index_vv=np.zeros([len(seg_labels)])
              sensitivity_vv=np.zeros([len(seg_labels)])
              specificity_vv=np.zeros([len(seg_labels)])
              f1_score_vv=np.zeros([len(seg_labels)])
              precision_vv=np.zeros([len(seg_labels)])
              pearscoeff_vv=np.zeros([len(seg_labels)])
              dice_score_vv=np.zeros([len(seg_labels)])
              if acc_flag :
                       for idx,value in enumerate(seg_labels): 
                           result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                           test_mask=(test_mask_all==value).astype(np.int)
                           try:
                               ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                               accuracy_vv[idx]=ConfMat.accuracy
                           except:
                               accuracy_vv[idx]= 0
                               
                       accuracy_v[i,:]=accuracy_vv
              if metrics_flag:
                                for idx,value in enumerate(seg_labels):
                                    result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                                    test_mask=(test_mask_all==value).astype(np.int)
                                    try:
                                        ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                                    
                                        
                                        MCC_vv[idx]=ConfMat.MCC
                                        Youden_index_vv[idx] = ConfMat.Youden_index
                                        sensitivity_vv[idx]  = ConfMat.sensitivity
                                        specificity_vv[idx]  = ConfMat.specificity
                                        f1_score_vv[idx]     = ConfMat.F1_score
                                        precision_vv[idx]    = ConfMat.precision
                                        rpea,ppea = scipy.stats.pearsonr(result_mask_seqeeze_int.flatten(), test_mask.flatten())
                                        pearscoeff_vv[idx]   = rpea
                                        dice_score_vv[idx]     = ConfMat.Dice_score
                        
                                    except:
                                        
                                        MCC_vv[idx]=0
                                        Youden_index_vv[idx] = 0
                                        sensitivity_vv[idx]  = 0
                                        specificity_vv[idx]  = 0
                                        f1_score_vv[idx]     = 0
                                        precision_vv[idx]    = 0
                                        rpea = 0
                                        pearscoeff_vv[idx]   = rpea
                                        dice_score_vv[idx]     = 0                                        
                        
                                MCC_v[i,:]          = MCC_vv
                                Youden_index_v[i,:] = Youden_index_vv
                                sensitivity_v[i,:]  = sensitivity_vv
                                specificity_v[i,:]  = specificity_vv
                                f1_score_v[i,:]     = f1_score_vv
                                precision_v[i,:]    = precision_vv                                
                                pearscoeff_v[i,:]   = pearscoeff_vv
                                dice_score_v[i,:]   = dice_score_vv
                    
              if plot_accuracy:
                           for idx,value in enumerate(seg_labels):
                               result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                               test_mask=(test_mask_all==value).astype(np.int)
                               try:
                                   ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                                   ConfMat.plot(),plt.suptitle("index: "+str(ix)+" -  label: "+str(value)+" - Accuracy: "+'%.3f' % ConfMat.accuracy + " - Dice Score: "+'%.3f' % ConfMat.Dice_score)
                               except:
                                    pass                                      

        self.accuracy =  accuracy_v
        self.accuracy_avg =  np.mean(accuracy_v,axis=0)   
        self.accuracy_std =  np.std(accuracy_v,axis=0)            
        if metrics_flag:          
              self.MCC_avg =  np.mean(MCC_v,axis=0)       
              self.MCC_std =  np.std(MCC_v,axis=0)
              self.Youden_index_avg =  np.mean(Youden_index_v,axis=0)
              self.Youden_index_std =  np.std(Youden_index_v,axis=0)
              self.sensitivity_avg  =  np.mean(sensitivity_v,axis=0)
              self.sensitivity_std  =  np.std(sensitivity_v,axis=0)
              self.specificity_avg  =  np.mean(specificity_v,axis=0)
              self.specificity_std  =  np.std(specificity_v,axis=0)
              self.pearscoeff_avg   =  np.mean(pearscoeff_v,axis=0)
              self.pearscoeff_std   =  np.std(pearscoeff_v,axis=0)
              self.f1_score_avg     =  np.mean(f1_score_v,axis=0)
              self.f1_score_std     =  np.std(f1_score_v,axis=0)
              self.precision_avg    =  np.mean(precision_v,axis=0)
              self.precision_std    =  np.std(precision_v,axis=0)
              self.dice_score_avg   =  np.mean(dice_score_v,axis=0)
              self.dice_score_std   =  np.std(dice_score_v,axis=0)

              self.MCC =  MCC_v
              self.Youden_index =  Youden_index_v
              self.sensitivity  =  sensitivity_v
              self.specificity  =  specificity_v
              self.pearscoeff   =  pearscoeff_v
              self.f1_score     =  f1_score_v
              self.precision    =  precision_v
              self.dice_score   =  dice_score_v


	      if print_metrics_flag:

	
		      print "Average metrics (between labels): "	
		      print 'Sensitivity (recall):    {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.sensitivity_avg),std_vall=np.std(self.sensitivity_avg))
		      print 'Specificity:             {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.specificity_avg),std_vall=np.std(self.specificity_avg))
		      print 'Accuracy:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.accuracy_avg),std_vall=np.std(self.accuracy_avg))
		      print 'Dice Score:              {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.dice_score_avg),std_vall=np.std(self.dice_score_avg))
		      print 'Precision:               {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.precision_avg),std_vall=np.std(self.precision_avg))
		      print 'F1 score:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.f1_score_avg),std_vall=np.std(self.f1_score_avg))
		      print 'MCC:                     {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.MCC_avg),std_vall=np.std(self.MCC_avg))
		      print 'Youden index:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.Youden_index_avg),std_vall=np.std(self.Youden_index_avg)) 
		      print 'Pearson coef:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.pearscoeff_avg),std_vall=np.std(self.pearscoeff_avg))
	
	      if print_separately_flag:

			for idx,value in enumerate(seg_labels):

			      print("Metrics label: "+ str(value))	
			      
			      print 'Sensitivity (recall):    {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.sensitivity_avg[idx],std_vall=self.sensitivity_std[idx])
			      print 'Specificity:             {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.specificity_avg[idx],std_vall=self.specificity_std[idx])
			      print 'Accuracy:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.accuracy_avg[idx],std_vall=self.accuracy_std[idx])
			      print 'Dice Score:              {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.dice_score_avg[idx],std_vall=self.dice_score_std[idx])
			      print 'Precision:               {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.precision_avg[idx],std_vall=self.precision_std[idx])              
			      print 'F1 score:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.f1_score_avg[idx],std_vall=self.f1_score_std[idx])
			      print 'MCC:                     {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.MCC_avg[idx],std_vall=self.MCC_std[idx])
			      print 'Youden index:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.Youden_index_avg[idx],std_vall=self.Youden_index_std[idx])
			      print 'Pearson coef:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.pearscoeff_avg[idx],std_vall=self.pearscoeff_std[idx])

        i=-1
        if plot_flag:
            
            for ix in indices: 
                 test_mask_i = Y_test[ix].astype(float)
                 test_mask = test_mask_i[:, :, :, 0]
		 i=i+1
                 OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Predicted Segmentation")
		 if plot_ground_truth:
		 	OrtoView(test_mask, colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Ground Truth Segmentation")	
                 result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i]) 
                 if plot_separately_flag:
                     for idx,value in enumerate(seg_labels):
                          result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                          test_mask=(test_mask_all==value).astype(np.int)                
                          OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("label: "+str(value)+" - Predicted mask")
                          OrtoView(test_image[i],Mask=test_mask), plt.suptitle("label: "+str(value)+"Ground Truth Mask superimposed on the Test Image")
                          OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Predicted Mask superimposed on the Test image")
                          OrtoView(test_mask, colormap=plt.get_cmap("Greens"),Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Predicted Mask (red) superimposed on Ground Truth Mask (green)")
            if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig

    def predict(self, X_test, **kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test[0].shape[0]
            self.IMG_HEIGHT = X_test[0].shape[1]
            self.IMG_LENGTH = X_test[0].shape[2]
            self.images = X_test
            self.image_type = X_test.dtype          
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        return_prob = kwargs.get('prob_masks',False)
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
              
        seg_labels=kwargs.get('labels',None)       
        plot_flag = kwargs.get('plot', False)
        plot_separately_flag=kwargs.get('plot_separately',False)
        
        test_image=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])
        result_mask_seqeeze_orig=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])
        result_mask_softmax = np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2],self.num_labels])
        i=-1    
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = test_image_i[:, :, :, 0]
  
              test_image_reshape = np.reshape(test_image[i], [-1, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)
              result_mask_softmax_i =  self.sess.run([tf.nn.log_softmax(self.logits, axis=-1)], feed_dict=test_data)
              result_mask_softmax[i]=np.array(result_mask_softmax_i).squeeze()
              result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i])            
              
              if seg_labels is None:
                  seg_labels=np.unique(result_mask_seqeeze_int_all)
                  
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

              
	i=-1
        if plot_flag:
            
            for ix in indices: 
		 i=i+1
                 OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Predicted Segmentation")
                 result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i]) 
                 if plot_separately_flag:
                     for idx,value in enumerate(seg_labels):
                          result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                          OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("label: "+str(value)+" result mask")
                          OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Test image vs  Result Mask")
        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        if return_prob:
            if result_mask_softmax.shape[0] == 1:
                result_mask_softmax=result_mask_softmax[0]
            return result_mask_softmax
        return result_mask_seqeeze_orig     

       


class UNET_3D_multiclass_v2(UNET_3D_multiclass):
	
    def __init__(self, **kwargs):
        
        self.gpu_num = kwargs.get('gpu_num', self.gpu_num)
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=self.gpu_num
        print("Use GPU device number "+self.gpu_num)
        with tf.device('/gpu:'+self.gpu_num):
              
	      
	      self.sess = tf.Session(config=self.config)

              self.loss_values = []   
              self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
              self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
              self.ckpt_step = kwargs.get('ckpt_step', None)
              self.ncores = kwargs.get('ncores', 1) 
              
              if self.ckpt_step is not None:
                    if self.ckpt_dir[-1] != "/":
                          self.ckpt_dir=self.ckpt_dir+"/"
                    checkpoint_file=self.ckpt_dir+"checkpoint"
                    if type(self.ckpt_step) is int:
                          self.ckpt_step=str(self.ckpt_step)
                    elif  type(self.ckpt_step) is str:
                          pass
                    else:
                         raise ValueError("ckpt_step must be string or int type") 
                    model_checkpoint_path="model_checkpoint_path: "+"\""+self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+"\""
                    all_model_checkpoint_paths="all_model_checkpoint_paths: "+"\""+self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+"\""
                    
                    
                    if not os.path.exists(self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+".meta"):
                          raise IOError(self.ckpt_dir+""+self.ckpt_basename+"-"+self.ckpt_step+" does not exists")
                    
                    
                    outF = open(checkpoint_file, "w")
                    for line in [model_checkpoint_path,all_model_checkpoint_paths]:
                          print >>outF, line
                    outF.close()
              self.towritedir = kwargs.get('towritedir', self.towritedir)        
              self.filter1stnumb = kwargs.get('filter1stnumb', 32)
              self.loss_type = kwargs.get('loss_type', "sparse_softmax_cross_entropy")
              self.dp_rate = kwargs.get('drop_rate', 0.75)
              self.drop_rate = tf.constant(self.dp_rate)
              self.drop_training = kwargs.get('drop_training', True)
              self.gstep = tf.Variable(0, dtype=tf.int32,
                                  trainable=False, name='global_step')
              self.use_softmax = kwargs.get('use_softmax', False)
              self.lr = tf.placeholder(tf.float32)
              self.learning_rate = 0.0001
              self.accuracy = 0.0
              self.dice = 0.0
    	      self.num_labels = kwargs.get('num_labels', self.num_labels) 	
              self.conv_kernel_size = kwargs.get('conv_kernel_size', self.conv_kernel_size) 
              self.batch_norm = kwargs.get('batch_norm', False) 
              self.dropout_layer = kwargs.get('dropout_layer', False) 
              self.print_shape = kwargs.get('print_shape', False) 
	
	
    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
			self.IMG_WIDTH = X[0].shape[0]
			self.IMG_HEIGHT = X[0].shape[1]
			self.IMG_LENGTH = X[0].shape[2]
              
			block1, conv2=full_block(X,self.filter1stnumb,
							   self.conv_kernel_size, 'SAME', tf.nn.relu,
							   self.pool_size,
							   'conv1', 'conv2', 'pool1',
							   self.batch_norm,
							   self.dropout_layer,
							   print_shape=self.print_shape,
							   drop_rate=self.drop_rate,drop_training=self.drop_training)
			block2, conv4=full_block(block1,self.filter1stnumb*2,
							   self.conv_kernel_size, 'SAME', tf.nn.relu,
							   self.pool_size,
							   'conv3','conv4', 'pool2',
							   self.batch_norm,
							   self.dropout_layer,
							   print_shape=self.print_shape,
							   drop_rate=self.drop_rate,drop_training=self.drop_training)
			block3, conv6=full_block(block2,self.filter1stnumb*4,
							   self.conv_kernel_size,
							   'SAME',tf.nn.relu,self.pool_size,
							   'conv5','conv6','pool3',
							   self.batch_norm,
							   self.dropout_layer,
							   print_shape=self.print_shape,
							   drop_rate=self.drop_rate,drop_training=self.drop_training)
			block4, conv8=full_block(block3,self.filter1stnumb*8,
							   self.conv_kernel_size,'SAME',tf.nn.relu,
							   self.pool_size,
							   'conv7','conv8','pool4',
							   self.batch_norm,
							   self.dropout_layer,
							   print_shape=self.print_shape,
							   drop_rate=self.drop_rate,drop_training=self.drop_training)
			block5 = conv3d_block(block4,self.filter1stnumb*16,
							   self.conv_kernel_size,'SAME',tf.nn.relu,
							   'conv9','conv10',
							   self.batch_norm,
							   print_shape=self.print_shape)             
			dropout = tf.layers.dropout(block5,
                                self.drop_rate,
                                training=self.drop_training,
                                name='dropout')
                                          
			dimm=[self.IMG_WIDTH / 8,self.IMG_HEIGHT / 8  ,self.IMG_LENGTH / 8 ]
			block6 = deconv3d_block(dropout,conv8,dimm,self.filter1stnumb*8, self.filter1stnumb*16,
							self.filter1stnumb*8,self.conv_kernel_size,'SAME',tf.nn.relu,
							"deconv1",'conv11', 'conv12',
							self.batch_norm,
							self.dropout_layer,
							self.print_shape,
							self.drop_rate,self.drop_training)

			dimm=[self.IMG_WIDTH / 4,self.IMG_HEIGHT / 4 ,self.IMG_LENGTH / 4 ]
			block7 = deconv3d_block(block6,conv6,dimm,self.filter1stnumb*4, self.filter1stnumb*8,
							self.filter1stnumb*4,self.conv_kernel_size,'SAME',tf.nn.relu,
							"deconv2",'conv13', 'conv14',
							self.batch_norm,
							self.dropout_layer,
							self.print_shape,
							self.drop_rate,self.drop_training) 

			dimm=[self.IMG_WIDTH / 2,self.IMG_HEIGHT / 2 ,self.IMG_LENGTH / 2 ]
			block8 = deconv3d_block(block7,conv4,dimm,self.filter1stnumb*2, self.filter1stnumb*4,
							self.filter1stnumb*2,self.conv_kernel_size,'SAME',tf.nn.relu,
							"deconv3",'conv15', 'conv16',
							self.batch_norm,
							self.dropout_layer,
							self.print_shape,
							self.drop_rate,self.drop_training) 

			dimm=[self.IMG_WIDTH ,self.IMG_HEIGHT ,self.IMG_LENGTH]
			block9 = deconv3d_block(block8,conv2,dimm,self.filter1stnumb*1, self.filter1stnumb*2,
							self.filter1stnumb,self.conv_kernel_size,'SAME',tf.nn.relu,
							"deconv4",'conv17', 'conv18',
							self.batch_norm,
							self.dropout_layer,
							self.print_shape,
							self.drop_rate,self.drop_training) 
     
			logits = tf.layers.conv3d(inputs=block9,
                                        filters=self.num_labels,
                                        kernel_size=[1, 1, 1],
                                        padding='SAME',
                                        activation=tf.identity,
                                        name='logits')
			if self.print_shape:
			  print("shape logits: "+str(logits.shape))    
			     
			return logits

class UNET_3D_multichannels(UNET_3D_multiclass):


    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]

              Xa=X[:,:,:,:,0,:];
              Xb=X[:,:,:,:,1,:]
              
              print("x shape: "+str(Xa.shape))

              conv1 = tf.layers.conv3d(inputs=Xa,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv3d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling3d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv3d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv3d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling3d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv3d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv3d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling3d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv3d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv3d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling3d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv3d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv3d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape)) 
#----------


              conv1b = tf.layers.conv3d(inputs=Xb,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1b')
              print("shape conv1b: "+str(conv1b.shape))
              conv2b = tf.layers.conv3d(inputs=conv1b,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2b')
              print("shape conv2b: "+str(conv2b.shape))
              
              pool1b = tf.layers.max_pooling3d(inputs=conv2b,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1b')
              print("shape pool1b: "+str(pool1b.shape))
              conv3b = tf.layers.conv3d(inputs=pool1b,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3b')
              print("shape conv3b: "+str(conv3b.shape))
              conv4b = tf.layers.conv3d(inputs=conv3b,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4b')
              print("shape conv4b: "+str(conv4b.shape))
              pool2b = tf.layers.max_pooling3d(inputs=conv4b,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2b')
              print("shape pool2b: "+str(pool2b.shape))
              conv5b = tf.layers.conv3d(inputs=pool2b,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5b')
              print("shape conv5: "+str(conv5.shape))
              conv6b = tf.layers.conv3d(inputs=conv5b,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6b')
              print("shape conv6b: "+str(conv6b.shape))
              pool3b = tf.layers.max_pooling3d(inputs=conv6b,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3b')
              print("shape pool3b: "+str(pool3b.shape))
              conv7b = tf.layers.conv3d(inputs=pool3b,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7b')
              print("shape conv7: "+str(conv7.shape))
              conv8b = tf.layers.conv3d(inputs=conv7b,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8b')
              print("shape conv8b: "+str(conv8b.shape))
              pool4b = tf.layers.max_pooling3d(inputs=conv8b,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4b')
              print("shape pool4b: "+str(pool4b.shape))
              conv9b = tf.layers.conv3d(inputs=pool4b,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9b')
              print("shape conv9b: "+str(conv9b.shape))
              conv10b = tf.layers.conv3d(inputs=conv9b,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10b')
              print("shape conv10b: "+str(conv10b.shape)) 

#----------
              conv10 = concatenate([conv10, conv10b]) 
              print("shape conacat conv10: "+str(conv10.shape))  

              conv10_plus = tf.layers.conv3d(inputs=conv10,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10_plus')
              print("shape conv10_plus: "+str(conv10_plus.shape)) 
            
              dropout = tf.layers.dropout(conv10_plus,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))
              deconv1 = deconv3d(dropout, 1, self.IMG_WIDTH / 8, self.filter1stnumb*8, self.filter1stnumb*16, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate([deconv1, conv8]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv11 = tf.layers.conv3d(inputs=deconv1,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              
              conv12 = tf.layers.conv3d(inputs=conv11,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv12')             
              print("shape conv12: "+str(conv12.shape))              
              deconv2 = deconv3d(conv12, 1,self.IMG_WIDTH / 4, self.filter1stnumb*4, self.filter1stnumb*8, "deconv2")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate([deconv2, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv13 = tf.layers.conv3d(inputs=deconv2,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv3d(inputs=conv13,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv14')
              print("shape conv14: "+str(conv14.shape))              
              deconv3 = deconv3d(conv14, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv3: "+str(deconv3.shape))
              deconv3 = concatenate([deconv3, conv4]) 
              print("shape deconv3 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv3d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              conv16 = tf.layers.conv3d(inputs=conv15,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv16')
              print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv3d(conv16, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate([deconv4, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv3d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              conv18 = tf.layers.conv3d(inputs=conv17,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv18')
              print("shape conv18: "+str(conv18.shape))       
              logits = tf.layers.conv3d(inputs=conv18,
                                        filters=self.num_labels,
                                        kernel_size=[1, 1, 1],
                                        padding='SAME',
                                        activation=tf.identity,
                                        name='logits')
              print("shape logits: "+str(logits.shape))        
              return logits
    

    
    def shuffle(self):
        p = np.random.permutation(len(self.X_train))
        self.images = self.X_train[p]
        self.labels = self.Y_train[p]
    
    def next_batch(self, iters):
        if (iters == 0):
            self.shuffle()
        count = self.batch_size * iters
        return self.images[count:(count + self.batch_size)], self.labels[count:(count + self.batch_size)]
    
    def initialize_XY(self,X_t,Y_t=None):

                  self.IMG_WIDTH = X_t[0].shape[0]
                  self.IMG_HEIGHT = X_t[0].shape[1]
                  self.IMG_LENGTH = X_t[0].shape[2]
                  self.IMG_CHANNEL = X_t[0].shape[3]

		  
                  Y_type=tf.float32
		  X_type=tf.float32
		  if X_t.dtype == np.float16:
			X_type=tf.float16
		  if Y_t is not None:
		  	if Y_t.dtype == np.float16:
				Y_type=tf.float16

		  if  self.IMG_CHANNEL != 1: 
                  	
                  	self.X = tf.placeholder(X_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH,self.IMG_CHANNEL, 1])
                  	if Y_t is not None:
                          self.Y_ = tf.placeholder(Y_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
                          self.labels = Y_t
                          self.label_type=Y_t.dtype
                  	self.dims=[self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH]
		  else:
                  	self.IMG_CHANNEL = self.IMG_LENGTH
                  	self.X = tf.placeholder(X_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_CHANNEL, 1])
                  	if Y_t is not None:
                          self.Y_ = tf.placeholder(Y_type, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
                          self.labels = Y_t
                          self.label_type=Y_t.dtype
                  	self.dims=[self.IMG_WIDTH, self.IMG_HEIGHT]
                  self.images  = X_t 
                  self.image_type = X_t.dtype
		          

    def train(self,X_train,Y_train, **kwargs):
        
        with tf.device('/gpu:'+self.gpu_num):  
            
            
              self.X_train = X_train
              self.Y_train = Y_train

              if self.logits is None:
                      self.initialize_XY(X_train,Y_train)
            
            
            
              self.iterations = kwargs.get('iter', self.iterations)
              self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
              self.batch_size = kwargs.get('batch_size', self.batch_size)
              self.batch_count_thr = kwargs.get('batch_count_thr', int(self.X_train.shape[0]/self.batch_size))
              self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
              self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
              self.skip_iter = kwargs.get('skip_iter', int(float(self.iterations*0.1)))      
              self.agument_batch_flag = kwargs.get('agument_batch', False) 
              self.parallel_transf = kwargs.get('parallel_transf', False) 
              if self.skip_iter == 0:
                    self.skip_iter=1
              if self.batch_count_thr == 0:
                    self.self.batch_count_thr=1        
              print("iterations: " + str(self.iterations))
              print("skip every "+str(self.skip_iter)+" iterations" )
              print("batch size: "+str(self.batch_size) )
              print("batch threshold: "+str(self.batch_count_thr) )
              
              if self.logits is None:
                    self.logits = self.build_model(self.X)
                    if self.use_softmax:
                          predicted=self.pixel_wise_softmax(self.logits)
                    else:
                          predicted=self.logits
                    if self.loss_type == "sigmoid_cross_entropy":
                          self.loss = tf.losses.sigmoid_cross_entropy(self.Y_,predicted)  
                    
                    elif self.loss_type == "softmax_cross_entropy":
                           self.loss = tf.losses.softmax_cross_entropy(tf.cast(self.Y_, tf.int32),predicted)

		    elif self.loss_type == "sparse_softmax_cross_entropy":
			   self.loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.Y_, tf.int32),predicted ) 
                                       
 		    elif self.loss_type == "softmax_cross_entropy_with_logits":
			   self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y_,logits=predicted )                         
                    
                    elif self.loss_type == "dice_score": 
                          self.loss = 1 - dice_coe(predicted,self.Y_,smooth=1e-8)
                    
                    elif self.loss_type == "mean_squared_error":                          
                          self.loss = tf.losses.mean_squared_error(self.Y_,predicted)
                    
                    elif self.loss_type == "focal_loss":
                          #self.loss = focal_loss(self.logits,tf.cast(self.Y_, tf.float32), gamma=2)

#                          self.loss = focal_loss_on_object_detection(self.Y_,predicted , gamma=2)
                          self.loss = focal_loss_on_object_detection(self.Y_,predicted , gamma=2)
                    
                    elif self.loss_type == "cosine_distance":
                          self.loss = tf.losses.cosine_distance(self.Y_,predicted,weights=1.0,axis=1)
                    
                    elif self.loss_type == "absolute_difference":
                          self.loss = tf.losses.absolute_difference(self.Y_,predicted,weights=1.0)
                    else:
                          raise NameError('Invalid Loss Function')
                    
                    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
              
              # Summary
              self.summary()
              
              # init
              init = tf.global_variables_initializer()
              
              self.sess.run(init)
              saver = tf.train.Saver()
              if self.ckpt_dir:
                  trymakedir(self.ckpt_dir)
                  ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
                  
                  if ckpt and ckpt.model_checkpoint_path:
                      saver.restore(self.sess, ckpt.model_checkpoint_path)
              try:
                  step = self.gstep.eval(session=self.sess)
              except:
                  step = 0
                  pass
              
              batch_count = 0
              writer = tf.summary.FileWriter(self.towritedir, tf.get_default_graph())
              
              for i in xrange(self.iterations):
                  # training on batches 
                  if (batch_count > self.batch_count_thr):
                      batch_count = 0
                  batch_X, batch_Y = self.next_batch(batch_count)
                  batch_X = StandData(batch_X, batch_X)
                  if self.agument_batch_flag:
                       batch_X_ag, batch_Y_ag = self.agument_batch(batch_X, batch_Y,parallel_transf=self.parallel_transf)
                       batch_X = StandData(batch_X_ag, batch_X)
                       batch_Y = batch_Y_ag
                  batch_X = np.expand_dims(np.concatenate([NormImages(batch_X[:,:,:,:,0,:]),NormImages(batch_X[:,:,:,:,1,:])],axis=4),axis=5)
                  batch_count += 1
                  feed_dict = {self.X: batch_X, self.Y_: batch_Y, self.lr: self.learning_rate}
                  loss_value, _,summaries = self.sess.run([self.loss, self.optimizer, self.summary_op], feed_dict=feed_dict)
                  writer.add_summary(summaries, global_step=step)
                  self.loss_values.append(loss_value)
                  if (i % self.skip_iter == 0):
                      print(str(step) + " training loss:", str(loss_value))
                      step += 1
                      if self.ckpt_dir:

                          saver.save(self.sess, self.ckpt_dir + self.ckpt_basename, step)
              writer.close()
              print("Done!")
    

    def test(self, X_test, Y_test,**kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test[0].shape[0]
            self.IMG_HEIGHT = X_test[0].shape[1]
            self.IMG_LENGTH = X_test[0].shape[2]
            self.IMG_CHANNEL = X_test[0].shape[3]
            self.images, self.labels = X_test, Y_test
            self.image_type = X_test.dtype
            self.label_type=Y_test.dtype
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH,self.IMG_CHANNEL, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.logits = self.build_model(self.X)
            #self.initialize_XY(X_test, Y_test)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
              
        seg_labels=kwargs.get('labels',None)
       

        if seg_labels is None:
                       
              test_mask_i = Y_test[0].astype(np.float32)
              test_mask_all = test_mask_i[:, :, :, 0].astype(np.int16)
              
              seg_labels=np.unique(test_mask_all)
                  
              if np.any(seg_labels==0.0):
                      seg_labels=np.delete(seg_labels, 0)
              print("labels found: "+str(seg_labels))
        else:                     
              if isinstance(seg_labels, list) :
                      seg_labels.sort()
              elif isinstance(seg_labels, np.ndarray):
                      np.sort(seg_labels)
              elif isinstance(seg_labels, int):               
                       seg_labels=[seg_labels]                 
              else:
                       seg_labels=[seg_labels]
       
        
        plot_flag = kwargs.get('plot', False)
        metrics_flag = kwargs.get('compute_metrics', True)
        print_metrics_flag = kwargs.get('print_metric', True)
        acc_flag = kwargs.get('compute_accuracy',True)
        plot_accuracy=kwargs.get('plot_accuracy',True)
        plot_separately_flag=kwargs.get('plot_separately',False)
        print_separately_flag=kwargs.get('print_separately',False)
        plot_ground_truth=kwargs.get('plot_ground_truth',False)
        if metrics_flag:
              acc_flag=True
        i=-1
        test_image=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2],X_test[0].shape[3]])
        result_mask_seqeeze_orig=np.zeros([len(indices),Y_test[0].shape[0],Y_test[0].shape[1],Y_test[0].shape[2]])
        dice_v=np.zeros([len(indices),len(seg_labels)])                       
        accuracy_v=np.zeros([len(indices),len(seg_labels)]) 
        if metrics_flag:
              MCC_v=np.zeros([len(indices),len(seg_labels)])
              Youden_index_v=np.zeros([len(indices),len(seg_labels)])
              sensitivity_v=np.zeros([len(indices),len(seg_labels)])
              specificity_v=np.zeros([len(indices),len(seg_labels)])
              pearscoeff_v=np.zeros([len(indices),len(seg_labels)])
              f1_score_v=np.zeros([len(indices),len(seg_labels)])
              precision_v=np.zeros([len(indices),len(seg_labels)])
              dice_score_v=np.zeros([len(indices),len(seg_labels)])
              
        for ix in indices:
              i=i+1
              
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = test_image_i[:, :, :, :, 0]
              
              test_mask_i = Y_test[ix].astype(np.float32)
              test_mask_all = test_mask_i[:, :, :, 0].astype(np.int16)
             
              if seg_labels is None:
                  seg_labels=np.unique(test_mask_all)
                  
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

              test_image_reshape = np.reshape(test_image[i], [-1, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH,self.IMG_CHANNEL,  1])
              
              test_data = {self.X: test_image_reshape}
              #result_mask = self.sess.run([self.logits], feed_dict=test_data)
	      result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)
      
	      #return result_mask

              result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i])
              accuracy_vv=np.zeros([len(seg_labels)])
              MCC_vv=np.zeros([len(seg_labels)])
              Youden_index_vv=np.zeros([len(seg_labels)])
              sensitivity_vv=np.zeros([len(seg_labels)])
              specificity_vv=np.zeros([len(seg_labels)])
              f1_score_vv=np.zeros([len(seg_labels)])
              precision_vv=np.zeros([len(seg_labels)])
              pearscoeff_vv=np.zeros([len(seg_labels)])
              dice_score_vv=np.zeros([len(seg_labels)])
              if acc_flag :
                       for idx,value in enumerate(seg_labels): 
                           result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                           test_mask=(test_mask_all==value).astype(np.int)
                           try:
                               ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                               accuracy_vv[idx]=ConfMat.accuracy
                           except:
                               accuracy_vv[idx]= 0
                               
                       accuracy_v[i,:]=accuracy_vv
              if metrics_flag:
                                for idx,value in enumerate(seg_labels):
                                    result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                                    test_mask=(test_mask_all==value).astype(np.int)
                                    try:
                                        ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                                    
                                        
                                        MCC_vv[idx]=ConfMat.MCC
                                        Youden_index_vv[idx] = ConfMat.Youden_index
                                        sensitivity_vv[idx]  = ConfMat.sensitivity
                                        specificity_vv[idx]  = ConfMat.specificity
                                        f1_score_vv[idx]     = ConfMat.F1_score
                                        precision_vv[idx]    = ConfMat.precision
                                        rpea,ppea = scipy.stats.pearsonr(result_mask_seqeeze_int.flatten(), test_mask.flatten())
                                        pearscoeff_vv[idx]   = rpea
                                        dice_score_vv[idx]     = ConfMat.Dice_score
                        
                                    except:
                                        
                                        MCC_vv[idx]=0
                                        Youden_index_vv[idx] = 0
                                        sensitivity_vv[idx]  = 0
                                        specificity_vv[idx]  = 0
                                        f1_score_vv[idx]     = 0
                                        precision_vv[idx]    = 0
                                        rpea = 0
                                        pearscoeff_vv[idx]   = rpea
                                        dice_score_vv[idx]     = 0                                        
                        
                                MCC_v[i,:]          = MCC_vv
                                Youden_index_v[i,:] = Youden_index_vv
                                sensitivity_v[i,:]  = sensitivity_vv
                                specificity_v[i,:]  = specificity_vv
                                f1_score_v[i,:]     = f1_score_vv
                                precision_v[i,:]    = precision_vv                                
                                pearscoeff_v[i,:]   = pearscoeff_vv
                                dice_score_v[i,:]   = dice_score_vv
                    
              if plot_accuracy:
                           for idx,value in enumerate(seg_labels):
                               result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                               test_mask=(test_mask_all==value).astype(np.int)
                               try:
                                   ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                                   ConfMat.plot(),plt.suptitle("index: "+str(ix)+" -  label: "+str(value)+" - Accuracy: "+'%.3f' % ConfMat.accuracy + " - Dice Score: "+'%.3f' % ConfMat.Dice_score)
                               except:
                                    pass                                      
        self.accuracy_avg =  np.mean(accuracy_v,axis=0)
        self.accuracy_std =  np.std(accuracy_v,axis=0)            
        if metrics_flag:          
              self.MCC_avg =  np.mean(MCC_v,axis=0)       
              self.MCC_std =  np.std(MCC_v,axis=0)
              self.Youden_index_avg =  np.mean(Youden_index_v,axis=0)
              self.Youden_index_std =  np.std(Youden_index_v,axis=0)
              self.sensitivity_avg  =  np.mean(sensitivity_v,axis=0)
              self.sensitivity_std  =  np.std(sensitivity_v,axis=0)
              self.specificity_avg  =  np.mean(specificity_v,axis=0)
              self.specificity_std  =  np.std(specificity_v,axis=0)
              self.pearscoeff_avg   =  np.mean(pearscoeff_v,axis=0)
              self.pearscoeff_std   =  np.std(pearscoeff_v,axis=0)
              self.f1_score_avg     =  np.mean(f1_score_v,axis=0)
              self.f1_score_std     =  np.std(f1_score_v,axis=0)
              self.precision_avg    =  np.mean(precision_v,axis=0)
              self.precision_std    =  np.std(precision_v,axis=0)
              self.dice_score_avg   =  np.mean(dice_score_v,axis=0)
              self.dice_score_std   =  np.std(dice_score_v,axis=0)

              self.MCC =  MCC_v
              self.Youden_index =  Youden_index_v
              self.sensitivity  =  sensitivity_v
              self.specificity  =  specificity_v
              self.pearscoeff   =  pearscoeff_v
              self.f1_score     =  f1_score_v
              self.precision    =  precision_v
              self.dice_score   =  dice_score_v


	      if print_metrics_flag:

	
		      print "Average metrics (between labels): "	
		      print 'Sensitivity (recall):    {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.sensitivity_avg),std_vall=np.std(self.sensitivity_avg))
		      print 'Specificity:             {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.specificity_avg),std_vall=np.std(self.specificity_avg))
		      print 'Accuracy:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.accuracy_avg),std_vall=np.std(self.accuracy_avg))
		      print 'Dice Score:              {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.dice_score_avg),std_vall=np.std(self.dice_score_avg))
		      print 'Precision:               {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.precision_avg),std_vall=np.std(self.precision_avg))
		      print 'F1 score:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.f1_score_avg),std_vall=np.std(self.f1_score_avg))
		      print 'MCC:                     {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.MCC_avg),std_vall=np.std(self.MCC_avg))
		      print 'Youden index:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.Youden_index_avg),std_vall=np.std(self.Youden_index_avg)) 
		      print 'Pearson coef:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=np.mean(self.pearscoeff_avg),std_vall=np.std(self.pearscoeff_avg))
	
	      if print_separately_flag:

			for idx,value in enumerate(seg_labels):

			      print("Metrics label: "+ str(value))	
			      
			      print 'Sensitivity (recall):    {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.sensitivity_avg[idx],std_vall=self.sensitivity_std[idx])
			      print 'Specificity:             {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.specificity_avg[idx],std_vall=self.specificity_std[idx])
			      print 'Accuracy:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.accuracy_avg[idx],std_vall=self.accuracy_std[idx])
			      print 'Dice Score:              {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.dice_score_avg[idx],std_vall=self.dice_score_std[idx])
			      print 'Precision:               {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.precision_avg[idx],std_vall=self.precision_std[idx])              
			      print 'F1 score:                {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.f1_score_avg[idx],std_vall=self.f1_score_std[idx])
			      print 'MCC:                     {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.MCC_avg[idx],std_vall=self.MCC_std[idx])
			      print 'Youden index:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.Youden_index_avg[idx],std_vall=self.Youden_index_std[idx])
			      print 'Pearson coef:            {mean_vall:.4f} (+/- {std_vall:.4f})'.format(mean_vall=self.pearscoeff_avg[idx],std_vall=self.pearscoeff_std[idx])
        i=-1
        if plot_flag:
            
            for ix in indices: 
                 test_mask_i = Y_test[ix].astype(float)
                 test_mask = test_mask_i[:, :, :, 0]
		 i=i+1
                 OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Predicted Segmentation")

		 if plot_ground_truth:
		 	OrtoView(test_mask, colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Ground Truth Segmentation")	
                 result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i]) 
                 if plot_separately_flag:
                     for idx,value in enumerate(seg_labels):
                          result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                          test_mask=(test_mask_all==value).astype(np.int)                
                          OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("label: "+str(value)+" - Predicted mask")
                          OrtoView(test_image[i,:,:,:,0],Mask=test_mask), plt.suptitle("label: "+str(value)+"Ground Truth Mask superimposed on the Test Image")
                          OrtoView(test_image[i,:,:,:,0],Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Predicted Mask superimposed on the Test image")
                          OrtoView(test_mask, colormap=plt.get_cmap("Greens"),Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Predicted Mask (red) superimposed on Ground Truth Mask (green)")
            if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig

    def predict(self, X_test, **kwargs):
        


        if self.logits is None:
            
            self.IMG_WIDTH = X_test[0].shape[0]
            self.IMG_HEIGHT = X_test[0].shape[1]
            self.IMG_LENGTH = X_test[0].shape[2]
            self.IMG_CHANNEL = X_test[0].shape[3]
            self.images = X_test
            self.image_type = X_test.dtype
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH,self.IMG_CHANNEL, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        return_prob = kwargs.get('prob_masks',False)
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
              
        seg_labels=kwargs.get('labels',None)       
        plot_flag = kwargs.get('plot', False)
        plot_separately_flag=kwargs.get('plot_separately',False)
        

        test_image=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2],X_test[0].shape[3]])
        result_mask_seqeeze_orig=np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]])
        result_mask_softmax = np.zeros([len(indices),X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2],self.num_labels])
        i=-1    
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = test_image_i[:, :, :, 0]
  
              test_image_reshape = np.reshape(test_image[i], [-1, self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH,self.IMG_CHANNEL,  1])
              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)
              result_mask_softmax_i =  self.sess.run([tf.nn.log_softmax(self.logits, axis=-1)], feed_dict=test_data)
              result_mask_softmax[i]=np.array(result_mask_softmax_i).squeeze()
              result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT,self.IMG_LENGTH, 1])
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i])            
              
              if seg_labels is None:
                  seg_labels=np.unique(result_mask_seqeeze_int_all)
                  
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

              
	i=-1
        if plot_flag:
            
            for ix in indices: 
		 i=i+1
                 OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("index: "+str(ix)+" Predicted Segmentation")
                 result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig[i]) 
                 if plot_separately_flag:
                     for idx,value in enumerate(seg_labels):
                          result_mask_seqeeze_int=(result_mask_seqeeze_int_all==value).astype(np.int)
                          OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("label: "+str(value)+" result mask")
                          OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("label: "+str(value)+" Test image vs  Result Mask")
        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        if return_prob:
            if result_mask_softmax.shape[0] == 1:
                result_mask_softmax=result_mask_softmax[0]
            return result_mask_softmax
        return result_mask_seqeeze_orig   



###### 2D ########



class UNET_2D(UNET_3D):

    conv_kernel_size=[3,3]
    pool_size=[2,2]


    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]
              conv1 = tf.layers.conv2d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv2d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv2d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv2d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling2d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv2d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv2d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling2d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv2d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv2d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling2d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv2d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv2d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape))              
              dropout = tf.layers.dropout(conv10,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))
              deconv1 = deconv2d(dropout, 1, self.IMG_WIDTH / 8, self.filter1stnumb*8, self.filter1stnumb*16, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate2d([deconv1, conv8]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv11 = tf.layers.conv2d(inputs=deconv1,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              
              conv12 = tf.layers.conv2d(inputs=conv11,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv12')             
              print("shape conv12: "+str(conv12.shape))              
              deconv2 = deconv2d(conv12, 1,self.IMG_WIDTH / 4, self.filter1stnumb*4, self.filter1stnumb*8, "deconv2")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate2d([deconv2, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv13 = tf.layers.conv2d(inputs=deconv2,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv2d(inputs=conv13,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv14')
              print("shape conv14: "+str(conv14.shape))              
              deconv3 = deconv2d(conv14, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv3 = concatenate2d([deconv3, conv4]) 
              print("shape deconv2 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv2d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              conv16 = tf.layers.conv2d(inputs=conv15,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv16')
              print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv2d(conv16, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate2d([deconv4, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv2d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              conv18 = tf.layers.conv2d(inputs=conv17,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv18')
              print("shape conv18: "+str(conv18.shape))       
              logits = tf.layers.conv2d(inputs=conv18,
                                        filters=1,
                                        kernel_size=[1, 1],
                                        padding='SAME',
                                        activation=tf.nn.sigmoid,
                                        name='logits')
              print("shape logits: "+str(logits.shape))     

              self.midmaps = conv10
              return logits 
                 


    def get_midmaps(self, X_test, **kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test.shape[0]
            self.IMG_HEIGHT = X_test.shape[1]
            self.images = X_test
            self.image_type = X_test.dtype          
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        test_image=np.zeros([X_test[0].shape[0],X_test[0].shape[1]])
        result_mask_seqeeze_orig=np.zeros([X_test[0].shape[0],X_test[0].shape[1]])
        test_image_i = X_test.astype(np.float32)
        test_image = test_image_i[:, :, 0]
        test_image_reshape = np.reshape(test_image, [-1, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
        test_data = {self.X: test_image_reshape}
        result_mask = self.sess.run([self.logits], feed_dict=test_data)
        Maps =  self.sess.run([self.midmaps], feed_dict=test_data)
        result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT, 1])
        result_mask_seqeeze_orig = result_mask.squeeze()
        result_mask_seqeeze = result_mask.squeeze().astype(np.float)
        result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig)
        return Maps[0],result_mask_seqeeze_int_all

class UNET_2D_multiclass(UNET_3D):

    conv_kernel_size=[3,3]
    pool_size=[2,2]


    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]
              conv1 = tf.layers.conv2d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv2d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv2d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv2d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling2d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv2d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv2d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling2d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv2d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv2d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling2d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv2d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv2d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape))              
              dropout = tf.layers.dropout(conv10,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))
              deconv1 = deconv2d(dropout, 1, self.IMG_WIDTH / 8, self.filter1stnumb*8, self.filter1stnumb*16, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate2d([deconv1, conv8]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv11 = tf.layers.conv2d(inputs=deconv1,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              
              conv12 = tf.layers.conv2d(inputs=conv11,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv12')             
              print("shape conv12: "+str(conv12.shape))              
              deconv2 = deconv2d(conv12, 1,self.IMG_WIDTH / 4, self.filter1stnumb*4, self.filter1stnumb*8, "deconv2")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate2d([deconv2, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv13 = tf.layers.conv2d(inputs=deconv2,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv2d(inputs=conv13,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv14')
              print("shape conv14: "+str(conv14.shape))              
              deconv3 = deconv2d(conv14, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv3 = concatenate2d([deconv3, conv4]) 
              print("shape deconv2 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv2d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              conv16 = tf.layers.conv2d(inputs=conv15,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv16')
              print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv2d(conv16, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate2d([deconv4, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv2d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              conv18 = tf.layers.conv2d(inputs=conv17,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv18')
              print("shape conv18: "+str(conv18.shape))       
              logits = tf.layers.conv2d(inputs=conv18,
                                        filters=2,
                                        kernel_size=[1, 1],
                                        padding='SAME',
                                        activation=tf.identity,
                                        name='logits')
              print("shape logits: "+str(logits.shape))     

              self.midmaps = conv10
              return logits 
                 


    def get_midmaps(self, X_test, **kwargs):
        


        if self.logits is None:
            self.IMG_WIDTH = X_test.shape[0]
            self.IMG_HEIGHT = X_test.shape[1]
            self.images = X_test
            self.image_type = X_test.dtype          
            self.X = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
            self.Y_ = tf.placeholder(tf.float32, [None, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        test_image=np.zeros([X_test[0].shape[0],X_test[0].shape[1]])
        result_mask_seqeeze_orig=np.zeros([X_test[0].shape[0],X_test[0].shape[1]])
        test_image_i = X_test.astype(np.float32)
        test_image = test_image_i[:, :, 0]
        test_image_reshape = np.reshape(test_image, [-1, self.IMG_WIDTH, self.IMG_HEIGHT, 1])
        test_data = {self.X: test_image_reshape}
        result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)

        Maps =  self.sess.run([self.midmaps], feed_dict=test_data)
        result_mask = np.reshape(np.squeeze(result_mask), [self.IMG_WIDTH, self.IMG_HEIGHT, 1])
        result_mask_seqeeze_orig = result_mask.squeeze()
        result_mask_seqeeze = result_mask.squeeze().astype(np.float)
        result_mask_seqeeze_int_all = integerize_seg(result_mask_seqeeze_orig)
        return Maps[0],result_mask_seqeeze_int_all

    def test(self, X_test, Y_test,**kwargs):
        


        if self.logits is None:

            self.initialize_XY( X_test, Y_test)

            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_test.shape[0]))
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
        
        plot_flag = kwargs.get('plot', False)
        metrics_flag = kwargs.get('compute_metrics', True)
        acc_flag = kwargs.get('compute_accuracy',True)
        plot_accuracy=kwargs.get('plot_accuracy',True)
        if metrics_flag:
              acc_flag=True
        i=-1
        images_dim=[len(indices)]+self.dims
        test_image=np.zeros(images_dim)
        result_mask_seqeeze_orig=np.zeros(images_dim)
        dice_v=np.zeros([len(indices)])                       
        accuracy_v=np.zeros([len(indices)]) 
        if metrics_flag:
              MCC_v=np.zeros([len(indices)])
              Youden_index_v=np.zeros([len(indices)])
              sensitivity_v=np.zeros([len(indices)])
              specificity_v=np.zeros([len(indices)])
              pearscoeff_v=np.zeros([len(indices)])
              f1_score_v=np.zeros([len(indices)])
              precision_v=np.zeros([len(indices)])
              dice_score_v=np.zeros([len(indices)])
        for ix in indices:
              i=i+1
              test_image_i = X_test[ix].astype(np.float32)
              test_image[i] = np.squeeze(test_image_i)
              test_mask_i = Y_test[ix].astype(np.float32)
              test_mask = np.squeeze(test_mask_i.astype(np.int16))
              rsp_dim= [-1] +self.dims +[ 1]
              rslt_dim = self.dims + [1]
              test_image_reshape = np.reshape(test_image[i],rsp_dim)
              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)
              result_mask = np.reshape(np.squeeze(result_mask), rslt_dim)
              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])
              
              
              if acc_flag or plot_accuracy:
                    
                    ConfMat = ConfusionMatrix(result_mask_seqeeze_int, test_mask)
                    accuracy_v[i]=ConfMat.accuracy 
              if metrics_flag:
                    
                    
                    MCC_v[i]=ConfMat.MCC
                    Youden_index_v[i] = ConfMat.Youden_index
                    sensitivity_v[i]  = ConfMat.sensitivity
                    specificity_v[i]  = ConfMat.specificity
                    f1_score_v[i]     = ConfMat.F1_score
                    precision_v[i]    = ConfMat.precision
                    rpea,ppea = scipy.stats.pearsonr(result_mask_seqeeze_int.flatten(), test_mask.flatten())
                    pearscoeff_v[i]   = rpea
                    self.dice = np.mean(dice_v) 
                    dice_score_v[i]     = ConfMat.Dice_score
              if plot_accuracy:
                          ConfMat.plot(),plt.suptitle("index: "+str(ix)+" - Accuracy: "+'%.3f' % ConfMat.accuracy)
                          if metrics_flag:
                                      plt.suptitle("index: "+str(ix)+" - Accuracy: "+'%.3f' % ConfMat.accuracy+" - Dice Score: "+'%.3f' % ConfMat.Dice_score)
        self.accuracy =  np.mean(accuracy_v)            
        
        
        
        if metrics_flag:          
              self.MCC =  np.mean(MCC_v)
              self.Youden_index =  np.mean(Youden_index_v)
              self.sensitivity  =  np.mean(sensitivity_v)
              self.specificity  =  np.mean(specificity_v)
              self.pearscoeff   =  np.mean(pearscoeff_v)
              self.f1_score     =  np.mean(f1_score_v)
              self.precision    =  np.mean(precision_v)
              self.dice_score   =  np.mean(dice_score_v)
              print 'Sensitivity (recall):    {}'.format(round(self.sensitivity,4))
              print 'Specificity:             {}'.format(round(self.specificity,4))
              print 'Accuracy:                {}'.format(round(self.accuracy,4))               
              print 'Dice Score:              {}'.format(round(self.dice_score,4))
              print 'Precision:               {}'.format(round(self.precision,4))               
              print 'F1 score:                {}'.format(round(self.f1_score,4))
              print 'MCC:                     {}'.format(round(self.MCC,4))
              print 'Youden index:            {}'.format(round(self.Youden_index,4)) 
              print 'Pearson coef:            {}'.format(round(self.pearscoeff,4))
        
        if plot_flag:
            i=-1
            for ix in indices: 
                  
                  i=i+1
                  test_mask_i = Y_test[ix].astype(float)
                  test_mask = np.squeeze(test_mask_i)
                  
                  
                  result_mask_seqeeze =result_mask_seqeeze_orig[i].astype(np.float32)
                  
                  result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])
                  if len(self.dims) == 3:
                      OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("Result Mask (orig)")
                      OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("result mask int")
                      OrtoView(test_image[i], Mask=result_mask_seqeeze_orig[i]), plt.suptitle("Test image vs  Result Mask (orig - overlay)")
                      OrtoView(test_image[i], Mask=result_mask_seqeeze), plt.suptitle("Test image vs  Result Mask (overlay)")   
                      OrtoView(test_mask, Mask=result_mask_seqeeze,colormap=plt.get_cmap("Greens"))
                      OrtoView(test_image[i],Mask=test_mask), plt.suptitle("Test image vs  Test Mask (GT)")
                      OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("Test image vs  Result Mask int")
                      OrtoView(test_mask, colormap=plt.get_cmap("Greens"),Mask=result_mask_seqeeze_int), plt.suptitle("Test Mask vs  Result Mask int")
                  else:
                      plt.figure(),plt.imshow(result_mask_seqeeze_orig[i], cmap=matplotlib.cm.jet), plt.suptitle("Result Mask (orig)")
                      plt.figure(),plt.imshow(test_image[i], cmap=plt.get_cmap("gray")), plot_overlay(result_mask_seqeeze_orig[i],alpha=0.5), plt.suptitle("Test image vs  Result Mask (overlay)") 
                      plt.figure(),plt.imshow(test_image[i], cmap=plt.get_cmap("gray")), plot_overlay(result_mask_seqeeze_int,alpha=0.5), plt.suptitle("Test image vs  Result Mask int (overlay)") 
        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig  
   
    def predict(self, X_input, **kwargs):
        
        if self.logits is None:
            
            self.initialize_XY( X_input)
            self.logits = self.build_model(self.X)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.ckpt_dir = kwargs.get('ckpt_dir', self.ckpt_dir)
            self.ckpt_basename = kwargs.get('ckpt_basename', self.ckpt_basename)
            print(self.ckpt_dir + self.ckpt_basename)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_dir + self.ckpt_basename))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        indices = kwargs.get('indices', range(X_input.shape[0]))
        if isinstance(indices, list) :
              indices.sort()
        elif isinstance(indices, np.ndarray):
              np.sort(indices)
        else:
              indices=[indices]
        
        plot_flag = kwargs.get('plot', False)

        i=-1
        rsp_dim= [-1] +self.dims +[ 1]
        rslt_dim = self.dims + [1]
        test_image_reshape = np.reshape(X_input[i],rsp_dim)
        images_dim=[len(indices)]+self.dims
        result_mask_seqeeze_orig=np.zeros(images_dim)    
        test_image=np.zeros(images_dim)
        for ix in indices:
              i=i+1
              test_image_i = X_input[ix].astype(np.float32)
              test_image[i] = np.squeeze(test_image_i)             
              test_image_reshape = np.reshape(test_image[i], rsp_dim)              
              test_data = {self.X: test_image_reshape}
              result_mask = self.sess.run([tf.argmax(tf.nn.log_softmax(self.logits, axis=-1), axis=-1)	], feed_dict=test_data)
              result_mask = np.reshape(np.squeeze(result_mask), rslt_dim)              
              result_mask_seqeeze_orig[i] = result_mask.squeeze()
              result_mask_seqeeze = result_mask.squeeze().astype(np.float)
              result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])
               
        

        if plot_flag:
            
            for ix in indices:        
                                  
                  i=i+1
                  result_mask_seqeeze =result_mask_seqeeze_orig[i].astype(np.float32)     
                  result_mask_seqeeze_int = integerize(result_mask_seqeeze_orig[i])                  
                  OrtoView(result_mask_seqeeze_orig[i], colormap=matplotlib.cm.jet), plt.suptitle("Result Mask (orig)")
                  OrtoView(result_mask_seqeeze_int, colormap=matplotlib.cm.jet), plt.suptitle("result mask int")
                  OrtoView(test_image[i], Mask=result_mask_seqeeze_orig[i]), plt.suptitle("Test image vs  Result Mask (orig - overlay)")
                  OrtoView(test_image[i], Mask=result_mask_seqeeze), plt.suptitle("Test image vs  Result Mask (overlay)")   
                  OrtoView(test_image[i],Mask=result_mask_seqeeze_int), plt.suptitle("Test image vs  Result Mask int")

        if result_mask_seqeeze_orig.shape[0] == 1:
              result_mask_seqeeze_orig=result_mask_seqeeze_orig[0]
        return result_mask_seqeeze_orig         


class UNET_2D_8blocks(UNET_2D_multiclass):

    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]

### Contracting Path

              conv1 = tf.layers.conv2d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              conv2 = tf.layers.conv2d(inputs=conv1,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv2')
              print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv2d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              conv4 = tf.layers.conv2d(inputs=conv3,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv4')
              print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling2d(inputs=conv4,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv2d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              conv6 = tf.layers.conv2d(inputs=conv5,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv6')
              print("shape conv6: "+str(conv6.shape))
              pool3 = tf.layers.max_pooling2d(inputs=conv6,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool3')
              print("shape pool3: "+str(pool3.shape))
              conv7 = tf.layers.conv2d(inputs=pool3,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv7')
              print("shape conv7: "+str(conv7.shape))
              conv8 = tf.layers.conv2d(inputs=conv7,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv8')
              print("shape conv8: "+str(conv8.shape))
              pool4 = tf.layers.max_pooling2d(inputs=conv8,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool4')
              print("shape pool4: "+str(pool4.shape))
              conv9 = tf.layers.conv2d(inputs=pool4,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv9')
              print("shape conv9: "+str(conv9.shape))
              conv10 = tf.layers.conv2d(inputs=conv9,
                                       filters=self.filter1stnumb*16,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv10')
              print("shape conv10: "+str(conv10.shape))  
              pool5 = tf.layers.max_pooling2d(inputs=conv10,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool5')
              print("shape pool5: "+str(pool5.shape))
              conv11 = tf.layers.conv2d(inputs=pool5,
                                       filters=self.filter1stnumb*32,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv11')
              print("shape conv11: "+str(conv11.shape))
              conv12 = tf.layers.conv2d(inputs=conv11,
                                       filters=self.filter1stnumb*32,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv12')
              print("shape conv12: "+str(conv12.shape))
              pool6 = tf.layers.max_pooling2d(inputs=conv12,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool6')
              print("shape pool6: "+str(pool6.shape))
              conv13 = tf.layers.conv2d(inputs=pool6,
                                       filters=self.filter1stnumb*64,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv13')
              print("shape conv13: "+str(conv13.shape))
              conv14 = tf.layers.conv2d(inputs=conv13,
                                       filters=self.filter1stnumb*64,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv14')
              print("shape conv14: "+str(conv14.shape))  
              pool7 = tf.layers.max_pooling2d(inputs=conv14,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool7')
              print("shape pool7: "+str(pool7.shape))
              conv15 = tf.layers.conv2d(inputs=pool7,
                                       filters=self.filter1stnumb*128,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv15')
              print("shape conv15: "+str(conv15.shape))
              conv16 = tf.layers.conv2d(inputs=conv15,
                                       filters=self.filter1stnumb*128,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv16')
              print("shape conv16: "+str(conv16.shape))  
              pool8 = tf.layers.max_pooling2d(inputs=conv16,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool8')
              print("shape pool8: "+str(pool8.shape))



#### Bottleneck  

              conv17 = tf.layers.conv2d(inputs=pool8,
                                       filters=self.filter1stnumb*256,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv17')
              print("shape conv17: "+str(conv17.shape))
              conv18 = tf.layers.conv2d(inputs=conv17,
                                       filters=self.filter1stnumb*256,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv18')
              print("shape conv18: "+str(conv18.shape))  
              dropout = tf.layers.dropout(conv18,
                                          self.drop_rate,
                                          training=self.drop_training,
                                          name='dropout')
              print("shape dropout: "+str(dropout.shape))

### Deconvolutional Path

              deconv1 = deconv2d(dropout, 1, self.IMG_WIDTH / 128, self.filter1stnumb*128, self.filter1stnumb*256, "deconv1")  
              print("shape deconv1: "+str(deconv1.shape))
              deconv1 = concatenate2d([deconv1, conv16]) 
              print("shape deconv1 concat: "+str(deconv1.shape))      
              
              conv19 = tf.layers.conv2d(inputs=deconv1,
                                       filters=self.filter1stnumb*128,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv19')
              print("shape conv19: "+str(conv19.shape))
              
              conv20 = tf.layers.conv2d(inputs=conv19,
                                        filters=self.filter1stnumb*128,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv20')             
              print("shape conv20: "+str(conv20.shape))   

##

              deconv2 = deconv2d(conv20, 1, self.IMG_WIDTH / 64, self.filter1stnumb*64, self.filter1stnumb*128, "deconv2")  
              print("shape deconv2: "+str(deconv2.shape))
              deconv2 = concatenate2d([deconv2, conv14]) 
              print("shape deconv2 concat: "+str(deconv2.shape))      
              
              conv21 = tf.layers.conv2d(inputs=deconv2,
                                       filters=self.filter1stnumb*64,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv21')
              print("shape conv21: "+str(conv21.shape))
              
              conv22 = tf.layers.conv2d(inputs=conv21,
                                        filters=self.filter1stnumb*64,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv22')             
              print("shape conv22: "+str(conv22.shape))              
              deconv3 = deconv2d(conv22, 1,self.IMG_WIDTH / 32, self.filter1stnumb*32, self.filter1stnumb*64, "deconv3")  # 32
              print("shape deconv3: "+str(deconv3.shape))
              deconv3 = concatenate2d([deconv3, conv12]) 
              print("shape deconv3 concat: "+str(deconv3.shape))
              conv23 = tf.layers.conv2d(inputs=deconv3,
                                        filters=self.filter1stnumb*32,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv23')
              print("shape conv13: "+str(conv23.shape))
              conv24 = tf.layers.conv2d(inputs=conv23,
                                        filters=self.filter1stnumb*32,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv24')
              print("shape conv14: "+str(conv24.shape))              
              deconv4 = deconv2d(conv24, 1, self.IMG_WIDTH / 16, self.filter1stnumb*16, self.filter1stnumb*32, "deconv4")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv4 = concatenate2d([deconv4, conv10]) 
              print("shape deconv2 concat: "+str(deconv3.shape))      
              conv25 = tf.layers.conv2d(inputs=deconv4,
                                        filters=self.filter1stnumb*16,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv25')
              print("shape conv25: "+str(conv25.shape)) 
              conv26 = tf.layers.conv2d(inputs=conv25,
                                        filters=self.filter1stnumb*16,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv26')
              print("shape conv26: "+str(conv26.shape))              
              deconv5 = deconv2d(conv26, 1, self.IMG_WIDTH/8, self.filter1stnumb*8, self.filter1stnumb*16, "deconv5")  # 32
              print("shape deconv5: "+str(deconv5.shape))
              deconv5 = concatenate2d([deconv5, conv8]) 
              print("shape deconv5 concat: "+str(deconv5.shape))  

###############

              conv27 = tf.layers.conv2d(inputs=deconv5,
                                       filters=self.filter1stnumb*8,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv27')
              print("shape conv27: "+str(conv27.shape))
              
              conv28 = tf.layers.conv2d(inputs=conv27,
                                        filters=self.filter1stnumb*8,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,      
                                        name='conv28')             
              print("shape conv28: "+str(conv28.shape))              
              deconv6 = deconv2d(conv28, 1,self.IMG_WIDTH / 4, self.filter1stnumb*4, self.filter1stnumb*8, "deconv6")  # 32
              print("shape deconv2: "+str(deconv2.shape))
              deconv6 = concatenate2d([deconv6, conv6]) 
              print("shape deconv2 concat: "+str(deconv2.shape))
              conv29 = tf.layers.conv2d(inputs=deconv6,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv29')
              print("shape conv29: "+str(conv29.shape))
              conv30 = tf.layers.conv2d(inputs=conv29,
                                        filters=self.filter1stnumb*4,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv30')
              print("shape conv30: "+str(conv30.shape))              
              deconv7 = deconv2d(conv30, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv7")  # 32
              print("shape deconv7: "+str(deconv7.shape))
              deconv7 = concatenate2d([deconv7, conv4]) 
              print("shape deconv7 concat: "+str(deconv7.shape))      
              conv31 = tf.layers.conv2d(inputs=deconv7,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv31')
              print("shape conv15: "+str(conv15.shape)) 
              conv32 = tf.layers.conv2d(inputs=conv31,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv32')
              print("shape conv32: "+str(conv32.shape))              
              deconv8 = deconv2d(conv32, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv8")  # 32
              print("shape deconv4: "+str(deconv8.shape))
              deconv8 = concatenate2d([deconv8, conv2]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv33 = tf.layers.conv2d(inputs=deconv8,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv33')
              print("shape conv33: "+str(conv33.shape)) 
              conv34 = tf.layers.conv2d(inputs=conv33,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv34')
              print("shape conv34: "+str(conv34.shape))       
              logits = tf.layers.conv2d(inputs=conv34,
                                        filters=2,
                                        kernel_size=[1, 1],
                                        padding='SAME',
                                        activation=tf.identity,
                                        name='logits')
              print("shape logits: "+str(logits.shape))       

              self.midmaps = conv18
              return logits 

                     
class UNET_3D_light(UNET_3D_multiclass):
    def build_model(self, X):
        with tf.device('/gpu:'+self.gpu_num):
              self.IMG_WIDTH = X[0].shape[0]
              self.IMG_HEIGHT = X[0].shape[1]
              self.IMG_LENGTH = X[0].shape[2]
              conv1 = tf.layers.conv3d(inputs=X,
                                       filters=self.filter1stnumb,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv1')
              print("shape conv1: "+str(conv1.shape))
              #conv2 = tf.layers.conv3d(inputs=conv1,
              #                         filters=self.filter1stnumb,
              #                         kernel_size=self.conv_kernel_size,
              #                         padding='SAME',
              #                         activation=tf.nn.relu,
              #                         name='conv2')
              #print("shape conv2: "+str(conv2.shape))
              
              pool1 = tf.layers.max_pooling3d(inputs=conv1,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool1')
              print("shape pool1: "+str(pool1.shape))
              conv3 = tf.layers.conv3d(inputs=pool1,
                                       filters=self.filter1stnumb*2,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv3')
              print("shape conv3: "+str(conv3.shape))
              #conv4 = tf.layers.conv3d(inputs=conv3,
              #                         filters=self.filter1stnumb*2,
              #                         kernel_size=self.conv_kernel_size,
              #                         padding='SAME',
              #                         activation=tf.nn.relu,
              #                         name='conv4')
              #print("shape conv4: "+str(conv4.shape))
              pool2 = tf.layers.max_pooling3d(inputs=conv3,
                                              pool_size=self.pool_size,
                                              strides=2,
                                              name='pool2')
              print("shape pool2: "+str(pool2.shape))
              conv5 = tf.layers.conv3d(inputs=pool2,
                                       filters=self.filter1stnumb*4,
                                       kernel_size=self.conv_kernel_size,
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       name='conv5')
              print("shape conv5: "+str(conv5.shape))
              #conv6 = tf.layers.conv3d(inputs=conv5,
              #                         filters=self.filter1stnumb*4,
              #                         kernel_size=self.conv_kernel_size,
              #                         padding='SAME',
              #                         activation=tf.nn.relu,
              #                         name='conv6')
              #dropout = tf.layers.dropout(conv6,
              #                            self.drop_rate,
              #                            training=self.drop_training,
              #                            name='dropout')
              #print("shape dropout: "+str(dropout.shape))              
              deconv3 = deconv3d(conv5, 1, self.IMG_WIDTH / 2, self.filter1stnumb*2, self.filter1stnumb*4, "deconv3")  # 32
              print("shape deconv3: "+str(deconv3.shape))
              deconv3 = concatenate([deconv3, conv3]) 
              print("shape deconv3 concat: "+str(deconv3.shape))      
              conv15 = tf.layers.conv3d(inputs=deconv3,
                                        filters=self.filter1stnumb*2,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv15')
              print("shape conv15: "+str(conv15.shape)) 
              #conv16 = tf.layers.conv3d(inputs=conv15,
              #                          filters=self.filter1stnumb*2,
              #                          kernel_size=self.conv_kernel_size,
              #                          padding='SAME',
              #                          activation=tf.nn.relu,
              #                          name='conv16')
              #print("shape conv16: "+str(conv16.shape))              
              deconv4 = deconv3d(conv15, 1, self.IMG_WIDTH, self.filter1stnumb, self.filter1stnumb*2, "deconv4")  # 32
              print("shape deconv4: "+str(deconv4.shape))
              deconv4 = concatenate([deconv4, conv1]) 
              print("shape deconv4 concat: "+str(deconv4.shape))      
              conv17 = tf.layers.conv3d(inputs=deconv4,
                                        filters=self.filter1stnumb,
                                        kernel_size=self.conv_kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='conv17')
              print("shape conv17: "+str(conv17.shape)) 
              #conv18 = tf.layers.conv3d(inputs=conv17,
              #                          filters=self.filter1stnumb,
              #                          kernel_size=self.conv_kernel_size,
              #                          padding='SAME',
              #                          activation=tf.nn.relu,
              #                          name='conv18')
              #print("shape conv18: "+str(conv18.shape))       
	      logits = tf.layers.conv3d(inputs=conv17,
                                        filters=self.num_labels,
                                        kernel_size=[1, 1, 1],
                                        padding='SAME',
                                        activation=tf.identity,
                                        name='logits')
              print("shape logits: "+str(logits.shape))       
              return logits                 
