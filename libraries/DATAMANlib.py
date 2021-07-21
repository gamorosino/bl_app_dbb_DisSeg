#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:02:00 2019
#########################################################################################################################
#########################################################################################################################
###################		            							                                        ###################
################### title:                      DATA MANIPULATION LIBRARY                             ###################
###################		           							                                        ###################
###################	 description:                                                                    ###################
###################	 version:	    0.2.2.1                                                          ###################
###################	 notes:	        need to Install: py modules - nibabel, multiprocessing,          ###################
###################                                               tqdm, skimage                       ###################
###################                                                                                   ###################             
###################    bash version:   tested on GNU bash, version 4.3.48                             ###################
###################                                                                                   ###################
################### autor: gamorosino                                                                 ###################
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
###################       update: minor changes                                                       ###################
#########################################################################################################################
#########################################################################################################################

@author: gamorosino
"""
import os
import numpy as np
import nibabel as nib
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from skimage.transform import resize as skresize

def trymakedir(path):
    try:
        os.mkdir(path)
    except:
        pass

def StandData(X, X_ref):
            
            X_orig_dtype=X.dtype
            if X_orig_dtype == np.float64 or X_orig_dtype == np.float32 or X_orig_dtype == np.float16:
                  pass
            else:
                  print("StandData: convert to float32")
                  X = X.astype(np.float32)
            X_ref = X_ref.astype(np.float32)
            result=(X - X_ref.mean()) / X_ref.std()
            return result.astype(X_orig_dtype)


def NormData(X):
            
            X_orig_dtype=X.dtype
            if X_orig_dtype == np.float64 or X_orig_dtype == np.float32 or X_orig_dtype == np.float16:
                  pass
            else:
                  #print("NormData: convert to float32")
                  X = X.astype(np.float32)  
            result=(X - X.min()) / (X.max() - X.min())
            if len(np.unique(result.astype(X_orig_dtype))) > 5:
                result=result.astype(X_orig_dtype)
            return result

def NormImages(X):
    for i in xrange(len(X)):
        X[i]=NormData(X[i])
    return X

def integerize(X):
      
      #return ( np.log(100*X)  > 0 ).astype(np.int)
      #return ( X  > 0 ).astype(np.int16)
      #return (( np.round(X.astype(float) / X.astype(float).max())) > 0).astype(np.int)
      return (( X.astype(float) / X.astype(float).max()) > 0.85).astype(np.int)

#      return (X>0).astype(np.int32)

def integerize_seg(X):
    
    return ( np.round(X) ).astype(np.int)

def loadArray(fullpath):
    file_ext = fullpath[fullpath.rfind('.'):]
    Array=[]
    if file_ext == ".gz" or file_ext == ".nii":
        
        NiiStructure = nib.load(fullpath)
        Array = NiiStructure.get_data()
    
    elif file_ext == ".npy":
        
        Array = np.load(fullpath)
    else:
        print "not valid extension"
    return Array

def saveArray_as( ArrayNii_tosave, file_path_fixed, file_path_tosave,  *args):


				image_fixed = nib.load(file_path_fixed)
				head_fixed = image_fixed.get_header()
				affine_fixed = image_fixed.get_affine()

				ArrayNii_tosave=np.nan_to_num(ArrayNii_tosave)

				nii_flipped = nib.Nifti1Image(ArrayNii_tosave, affine=affine_fixed, header=head_fixed)
				nii_flipped.to_filename(file_path_tosave)




def fbasename(path):
        file_ext =  path[path.rfind('.'):]
        if file_ext == ".gz":
              file_name =  path[0:path.rfind('.')]
              file_name =  file_name[0:file_name.rfind('.')]
        else:
              file_name =  path[0:path.rfind('.')]    
        return file_name


def close_pool(pool):
       pool.close()
       pool.terminate()
       pool.join()     



def get_data_pool(train_ids,mask_ids,PATH_x,PATH_y,dims,squeeze,i):
        id_=train_ids[i]
        path = PATH_x + id_
        
        fbasename_img=fbasename(id_)
        fbasename_img_num=fbasename_img[fbasename_img.rfind("_")+1:]
        
        if PATH_y is not  None:
            mask_path=PATH_y + mask_ids[i]
            fbasename_mask=fbasename(mask_ids[i])
            fbasename_mask_num=fbasename_mask[fbasename_mask.rfind("_")+1:]
        
            if not fbasename_img_num == fbasename_mask_num:
                  raise ValueError(" mismatch between "+path+ " and "+mask_path)
        else:
            mask_=None
        ldims=list(dims)+[1]
        img = np.zeros(ldims, dtype=np.float32)      
        print("load img: "+path)
        try:
                    
                    img = loadArray(path)
                    if squeeze:
			img = np.squeeze(img)
        except:
              print("error with "+path )
              #raise ValueError(" Cannot work out file type of"+path)
        img_max=img.max();
        img_min=img.min();
        img=NormData(img);
        if len(np.unique(img)) < 5:
            print("warning: n° unique voxel < 5")
            return np.zeros_like(img), np.zeros_like(img)
        img = np.expand_dims(skresize( img , dims, mode='constant',order=0), axis=-1)
        if PATH_y is not  None:
            mask_ = np.zeros(ldims, dtype=np.float32)
            print("load labels: "+mask_path)
            try:
                        mask_ = loadArray(mask_path)
                        if squeeze:
                            np.squeeze(mask_)
            except:
                  print("error with "+mask_path )
              #raise ValueError(" Cannot work out file type of"+mask_path)
            mask_max=mask_.max();
            mask_min=mask_.min();
            mask_ = np.expand_dims(skresize(NormData(mask_), dims, mode='constant',order=0), axis=-1)
            mask_=np.round(mask_*(mask_max-mask_min)+mask_min);
            for idx,value in enumerate(np.unique(mask_)):
    			    mask_[mask_==value]=idx
            mask_.astype(np.float32)
                    
        if image_type == np.float64 or image_type == np.float32 or image_type == np.float16:
				  pass
        else:
				  img=img*(img_max-img_min)+img_min;


        
        return img.astype(np.float32),mask_



def get_data(PATH_x,PATH_y, image_type_, label_type_,dims,squeeze=False,ncores=1):

    global image_type
    image_type=image_type_
    label_type=label_type_
    
    if ncores == 1:

	    # Get train and test IDs
	    train_ids = next(os.walk(PATH_x))[2]
	    train_ids.sort()
	    ldims=[len(train_ids)]+list(dims)+[1]
	    if PATH_y is not None:
        	    mask_ids = next(os.walk(PATH_y))[2]
        	    mask_ids.sort()
        	    labels = np.zeros(ldims, dtype=label_type)
	    # Get and resize train images and masks
	    images = np.zeros(ldims, dtype=image_type)
	    
	    print('Getting and resizing train images and masks ... ')
	    #sys.stdout.flush()
	    for i, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
		
			id_=train_ids[i]
			path = PATH_x + id_
			
			fbasename_img=fbasename(id_)
			fbasename_img_num=fbasename_img[fbasename_img.rfind("_")+1:]
            if PATH_y is not None:
						mask_path=PATH_y + mask_ids[i]
						fbasename_mask=fbasename(mask_ids[i])
						fbasename_mask_num=fbasename_mask[fbasename_mask.rfind("_")+1:]
						if not fbasename_img_num == fbasename_mask_num:
												raise ValueError(" mismatch between "+path+ " and "+mask_path)
            img = np.zeros(dims, dtype=np.float32)
            img=np.expand_dims(img,axis=-1)          
            
            
			#print("load img: "+path)
            try:
				    
				    img = loadArray(path)
				    if squeeze:
				     img = np.squeeze(img)
            except:
			      print("error with "+path )
			      #raise ValueError(" Cannot work out file type of"+path)
            img_max=img.max();
            img_min=img.min();
            img=NormData(img);
            if len(np.unique(img)) < 5:
			    print("warning: n° unique voxel < 5")
			    return np.zeros_like(img), np.zeros_like(img)
            img = np.expand_dims(skresize( img , dims, mode='constant',order=0), axis=-1)
            images[i] = img.astype(image_type)
            if PATH_y is not None:            
                        mask_ = np.zeros(dims, dtype=np.float32)
                        mask_=np.expand_dims(mask_,axis=-1)
                        try:
            				    mask_ = loadArray(mask_path)
            				    if squeeze:
            				     mask_ = np.squeeze(mask_)
                        except:
            			      print("error with "+mask_path )
            			      #raise ValueError(" Cannot work out file type of"+mask_path)
			if image_type == np.float64 or image_type == np.float32 or image_type == np.float16:
				  pass
			else:
				  img=img*(img_max-img_min)+img_min;
                  
            if PATH_y is not None:                  
             mask_max=mask_.max();
             mask_min=mask_.min();
             mask_ = np.expand_dims(skresize(NormData(mask_), dims, mode='constant',order=0), axis=-1)
             mask_=np.round(mask_*(mask_max-mask_min)+mask_min);
             for idx,value in enumerate(np.unique(mask_)):
			    mask_[mask_==value]=idx   
#            labels[i] =np.expand_dims( mask_final, axis=-1)	    
             labels[i]=mask_
	    X = images
	    if PATH_y is not None:        
			Y = labels
	    else:
	    	    Y=None
                
	    print('get_data: Done!')
	    return X.astype(image_type), Y.astype(label_type)
    else:
   
      
	      train_ids = next(os.walk(PATH_x))[2]
	      if PATH_y is not None:
	      	      mask_ids = next(os.walk(PATH_y))[2]
	      	      mask_ids.sort()
	      else:
                 mask_ids = None       
	      train_ids.sort()

	      func = partial(get_data_pool,train_ids,mask_ids,PATH_x,PATH_y,dims,squeeze)
	      p = Pool(ncores)
	      MAP = p.map(func,range(len(train_ids)))
          
          
	      X = np.zeros(dims, dtype=np.float32)
	      X=np.expand_dims(X,axis=-1)	      
	      X=np.expand_dims(X,axis=0)
	      if PATH_y is not None:
                 Y = np.zeros(dims, dtype=np.float32)
                 Y=np.expand_dims(Y,axis=-1)	      
                 Y=np.expand_dims(Y,axis=0)

	      
	      for mapped in tqdm(MAP):
		          mo=mapped[0]
		          if PATH_y is not None:                  
		           m1=mapped[1]
		           m1[np.isnan(m1)]=0
		           condition2=np.count_nonzero(m1) > 0
		          else:
		           condition2=True
		          mo[np.isnan(mo)]=0		                            
		          if np.count_nonzero(mo) > 0 and condition2:                        
		                mo=np.expand_dims(mo, axis=0)
		                if PATH_y is not None:
		                 m1=np.expand_dims(m1, axis=0)
		                 Y=np.concatenate([Y, m1],axis=0)
		                X=np.concatenate([X, mo],axis=0)
                        	      
	      close_pool(p)
	      if PATH_y is not None:
	       Y_out=Y[1:].astype(label_type)
	      else:
	       Y_out = None
	      return  X[1:].astype(image_type), Y_out




