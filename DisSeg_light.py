#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:41:42 2021


#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
###################      title:            3D UNET for Brain Tissues Segmentation                     ###################
###################                                                                                   ###################
###################	 description:                                                                     ###################
###################	 version:     0.5.1.0.0                                                           ###################
###################	 notes:	        need to Install: py modules - tf,matplotlib,nibabel,skimage       ###################
###################                                                                                   ###################
###################                                                                                   ###################
###################                                                                                   ###################             
###################    bash version:   tested on GNU bash, version 4.3.48                             ###################
###################                                                                                   ###################
################### autor: gamorosino                                                                 ###################
################### email: g.amorosino@gmail.com                                                      ###################
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


@author: gamorosino


"""


#%%
########################################################################################################################
#########################################       Import Modules       ###################################################
########################################################################################################################


import os
import sys
import  getopt
import numpy as np
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
from os.path import isfile
import glob
import nibabel as nib
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rc('figure', max_open_warning = 0)
from libraries.UNET import UNET_3D_light
#focal_loss,focal_loss_sigmoid_on_2_classification,focal_loss_sigmoid_on_multi_classification,focal_loss_on_object_detection
from libraries.DATAMANlib import trymakedir,get_data,loadArray,NormImages,StandData,data_partition,data_split,load_pickle,save_pickle,integerize_seg,NormData, skresize
#%%
########################################################################################################################
#####################################        Variables Declaration      ################################################
########################################################################################################################


image_type_str='float32'
label_type_str='float32'

loss_type="sparse_softmax_cross_entropy"



drop_rate=0.15
iterations =  100  # 10000000  # 1000000
btch_s = 1
total_iter=10000000
skp_iter=iterations/10
#btch_cnt_thr = 50
conv_kernel_size=[3,3,3]
default_num_labels=7
#

trnngset_div=9      
validation_portion=0.10

use_softmax=False
#Dataset Flags
renovate_trainingset=False
renovate_testset=False
renovate_stats=False
renovate_metrics=False

# Net Flags
initialize_flag=True
train_flag=False
validate_flag=True
test_flag=False
predict_flag=False

#Agument batch
agument_flag=False

#Metrics Flags
compute_metrics_flag=True



tb_port=6060

if image_type_str == label_type_str:
      type_str=image_type_str
else:
      type_str=image_type_str+"_"+label_type_str

iamge_type=np.dtype(image_type_str)
label_type=np.dtype(label_type_str)

TRAIN_PATH_x=None
TRAIN_PATH_y=None
TEST_PATH_x=None
TEST_PATH_y=None


#%%
########################################################################################################################
#####################################               Functions             ##############################################
########################################################################################################################

def load_nib(T1_file):    
    T1_Struct=nib.load(T1_file);
    T1_aff=T1_Struct.get_affine(); 
    T1_header=T1_Struct.get_header();
    T1_img = T1_Struct.get_data();
    return T1_img,T1_header,T1_aff
    
def tf_resp(T1_img,dims):    
    img=NormData(T1_img);                                
    img = np.expand_dims(skresize( img ,dims, mode='constant',order=0), axis=-1);
    img = np.expand_dims(img , axis=0);
    return img    

def input_parsing(argv):
   #Input Parsing
   gpu_num=str(0)
   PATH_x=None
   PATH_y=None
   model_dir=None 
   train_flag=False
   test_flag=False   
   predict_flag=False
   renovate_traningset=False
   renovate_testset=False
   checkpoint_dir=None
   checkpoint_basename=None
   checkpoint_step=None#8012;#304048
   output=None
   btch_s=1
   trnngset_div=1
   cores=2
   num_labels=None
   keep_path_list=False
   fltr1stnmb=12 #32
   divis=0.5
   IMG_WIDTH = int(float(128 / divis))
   IMG_HEIGHT = int(float(128 / divis))
   IMG_LENGTH = int(float(128 / divis))  
   IMG_CHANNELS = 1 
   dims = (IMG_WIDTH,IMG_HEIGHT,IMG_LENGTH)  
   try:
      opts, args = getopt.getopt(argv,"htperaki:l:g:m:u:b:d:c:q:s:j:n:o:f:z:",["training","predict","test","re-trainingset","re-test","--keep-pathlist","images=","labels=","gpunum=","model-dir=","batch-size=","training-div=","checkpoint-dir=","checkpoint-basename=","checkpoint-step=","num-threads=","num-classes=","output=","num-1stfilter=","dims="])
   except getopt.GetoptError:
      print('error:')
      print sys.argv[0]+' [--trainig | --test | --predict] [-k] --images <path> [--labels <path>] [-m <path>] [-o <path>] [-g <num>] [-b <num>] [-d <num>] [-c <path>] [ -q <path>] [ -s <str> ] [ -j <num> ] [ -n <num> ]'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print sys.argv[0]+' [--trainig | --test | --predict] [-k] --images <path> [--labels <path>] [-m <path>] [-o <path>]  [-g <num>] [-b <num>] [-d <num>] [-c <path>] [ -q <str>] [ -s <num> ] [ -j <num> ] [ -n <num> ]'
         sys.exit()
      elif opt in ("-t" , "--training"):
          train_flag=True
      elif opt in ("-e" , "--test"):
          test_flag=True
      elif opt in ("-p" , "--predict"):
          predict_flag=True
      elif opt in ("-r" , "--re-trainingset"):
          renovate_traningset=True
      elif opt in ("-a" , "--re-test"):
          renovate_testset=True
      elif opt in ("-i", "--images"):
         PATH_x = arg
      elif opt in ("-l", "--labels"):
         PATH_y = arg
      elif opt in ("-m", "--model-dir"):
         model_dir = arg
      elif opt in ("-o", "--output"):
         output = arg
      elif opt in ("-g", "--gpunum"):
         gpu_num = str(arg);
      elif opt in ("-b", "--batch-size"):
         btch_s = int(arg)
      elif opt in ("-d", "--trainingset-div"):
         trnngset_div = int(arg)
      elif opt in ("-c", "--checkpoint-dir"):
         checkpoint_dir = arg
      elif opt in ("-q", "--checkpoint-basename"):
         checkpoint_basename = arg
      elif opt in ("-s", "--checkpoint-step"):
         checkpoint_step = int(arg)
      elif opt in ("-j", "--num-threads"):
         cores = int(arg)
      elif opt in ("-n", "--num-classes"):
         num_labels = int(arg)
      elif opt in ("-f", "--num-1stfilter"):
         print(int(arg))
         fltr1stnmb = int(arg)
      elif opt in ("-z", "--dims"):
         print(int(arg))
         try:
			shape_ = int(arg)
			IMG_WIDTH = shape_
			IMG_HEIGHT = shape_
			IMG_LENGTH = shape_    
			dims = (shape_,shape_,shape_)    
         except:
			dims = tuple(np.array(list(arg.replace(',','').replace('[','').replace(']','').replace(')','').replace('(',''))).astype(int))
			IMG_WIDTH = dims[0]
			IMG_HEIGHT = dims[1]
			IMG_LENGTH = dims[2]   
      elif opt in ("-k", "--keep-pathlist"):
         keep_path_list = True
   if (test_flag is False) and  (train_flag is False) and (predict_flag is False):
          print("specify at least one of the following option: --training|-t or --test|-e or --predict|-p")
          sys.exit()  

   return train_flag,test_flag,predict_flag,PATH_x,PATH_y,model_dir,gpu_num,renovate_traningset,renovate_testset,btch_s,trnngset_div,checkpoint_dir,checkpoint_basename,checkpoint_step, cores,output,num_labels,keep_path_list,fltr1stnmb,dims

def get_trainingset(TRAIN_PATH_x,TRAIN_PATH_y,renovate_trainingset,keep_path_list,trnngset_div,train_i,validate_flag,ext_image_filename,ext_label_filename,dims,ncores=1):
	
          X_train_files=[]
          Y_train_files=[]
          X_train = None
          Y_train = None
          X_train_files = []
          Y_train_files = []
          X_validate = None 
          Y_validate = None
          X_validate_file = None
          Y_validate_file = None
          TRAIN_PATH_x_list = None
          TRAIN_PATH_y_list = None
          VALID_PATH_x_list = None 
          VALID_PATH_y_list = None
          
          if trnngset_div==1:
            X_train_files=[save_dir+"X_train"+"_"+ext_image_filename+".npy"]
            Y_train_files=[save_dir+"Y_train"+"_"+ext_label_filename+".npy"]
          else:
            for nn in range(trnngset_div):
                X_train_files.append(save_dir+"X_train"+str(nn)+"_"+ext_image_filename+".npy")
                Y_train_files.append(save_dir+"Y_train"+str(nn)+"_"+ext_label_filename+".npy")
          
          X_train_file = X_train_files[train_i]
          Y_train_file = Y_train_files[train_i]
          
          if validate_flag:
            X_validate_file=save_dir+"X_validate"+"_"+ext_image_filename+".npy"
            Y_validate_file=save_dir+"Y_validate"+"_"+ext_label_filename+".npy"
          else:
			X_validate_file=None
			Y_validate_file=None
            
          #RENOVATE OR RESTORE TRAININGSET
          
          TRAIN_PATH_x_list_file=save_dir+'/TRAIN_PATH_x_list.pkl' 
          TRAIN_PATH_y_list_file=save_dir+'/TRAIN_PATH_y_list.pkl'
          
          if not ( isfile(TRAIN_PATH_x_list_file) and isfile(TRAIN_PATH_y_list_file) ):
                renovate_trainingset=True
                
          if keep_path_list:
				  
				  if ( isfile(TRAIN_PATH_x_list_file) and isfile(TRAIN_PATH_y_list_file) ) :
						print('loading stored training image paths list from: '+TRAIN_PATH_x_list_file)              
						TRAIN_PATH_x_list=load_pickle(TRAIN_PATH_x_list_file)           
						print('loading stored training label paths list from: '+TRAIN_PATH_y_list_file)
						TRAIN_PATH_y_list=load_pickle(TRAIN_PATH_y_list_file) 
				  else:
						keep_path_list  = False
						TRAIN_PATH_x_list=glob.glob(TRAIN_PATH_x+'/*')
						TRAIN_PATH_x_list.sort()
						TRAIN_PATH_y_list=glob.glob(TRAIN_PATH_y+'/*')
						TRAIN_PATH_y_list.sort()
          else:       
				TRAIN_PATH_x_list=glob.glob(TRAIN_PATH_x+'/*')
				TRAIN_PATH_x_list.sort()
				TRAIN_PATH_y_list=glob.glob(TRAIN_PATH_y+'/*')
				TRAIN_PATH_y_list.sort()    
                
          if not ( isfile(X_train_file) and isfile(Y_train_file) ):
                renovate_trainingset=True
                
          if validate_flag: 
              VALID_PATH_x_list_file=save_dir+'/VALID_PATH_x_list.pkl' 
              VALID_PATH_y_list_file=save_dir+'/VALID_PATH_y_list.pkl'
              if not ( isfile(X_validate_file) and isfile(Y_validate_file) ):
                  renovate_trainingset=True
              if  ( not renovate_trainingset or  keep_path_list ):
				  if ( isfile(VALID_PATH_x_list_file) and isfile(VALID_PATH_y_list_file) ) :
					print('loading stored validation image paths list from: '+VALID_PATH_x_list_file)              
					VALID_PATH_x_list=load_pickle(VALID_PATH_x_list_file)   
					print('loading stored validation label paths list from: '+VALID_PATH_y_list_file)
					VALID_PATH_y_list=load_pickle(VALID_PATH_y_list_file)    
				  else:
				    keep_path_list  = False                  
			  
          if renovate_trainingset:
                print("renovate the dataset...")
                if not keep_path_list:
					if validate_flag:
						TRAIN_PATH_x_list,TRAIN_PATH_y_list,VALID_PATH_x_list,VALID_PATH_y_list=data_split(TRAIN_PATH_x_list,TRAIN_PATH_y_list,validation_portion)
						save_pickle(VALID_PATH_x_list_file, VALID_PATH_x_list)
						save_pickle(VALID_PATH_y_list_file, VALID_PATH_y_list)                
					save_pickle(TRAIN_PATH_x_list_file, TRAIN_PATH_x_list)
					save_pickle(TRAIN_PATH_y_list_file, TRAIN_PATH_y_list)
                
                TRAIN_PATH_x,TRAIN_PATH_y = data_partition(TRAIN_PATH_x_list,TRAIN_PATH_y_list,train_i,trnngset_div)
                #print  TRAIN_PATH_x
                #print  TRAIN_PATH_y
                start_time_load = time.time()
                X_train, Y_train = get_data(TRAIN_PATH_x,TRAIN_PATH_y, iamge_type,label_type,dims,ncores=cores)
                X_train = StandData(X_train, X_train)
                Y_train=Y_train.astype(label_type)
                elapsed_time = time.time() - start_time_load
                print("time for loading data"+" : "+'%.3f' % elapsed_time+" sec")
                print('saving current images of trainingset as '+X_train_file)
                np.save(X_train_file,X_train)
                print('saving current labels of trainingset as '+Y_train_file)
                np.save(Y_train_file,Y_train)     
                
                if validate_flag:
                    start_time_load = time.time()
                    X_validate, Y_validate = get_data(VALID_PATH_x_list,VALID_PATH_y_list, iamge_type,label_type,dims,ncores=cores)
                    X_validate = StandData(X_validate, X_train)
                    Y_validate=Y_validate.astype(label_type)
                    elapsed_time = time.time() - start_time_load
                    print("time for loading data"+" : "+'%.3f' % elapsed_time+" sec")
                    print('saving current images of validation set as'+X_validate_file)
                    np.save(X_validate_file,X_validate)
                    print('saving current labels of validation set as'+Y_validate_file)
                    np.save(Y_validate_file,Y_validate)                      
          else:
                print("restore last trainingset...")
                print('loading stored training images from: '+X_train_file) 
                X_train=loadArray(X_train_file)
                print('loading stored training labels from: '+X_train_file) 
                Y_train=loadArray(Y_train_file)
                if validate_flag:
                    print("restore last validation set...")
                    print('loading stored validation images from: '+X_validate_file)
                    X_validate=loadArray(X_validate_file) 
                    print('loading stored validation labels from: '+Y_validate_file) 
                    Y_validate=loadArray(Y_validate_file)
          if validate_flag:          
			  occurrences_x=[i for i in VALID_PATH_x_list if i in TRAIN_PATH_x_list ]          
			  occurrences_y=[i for i in VALID_PATH_y_list if i in TRAIN_PATH_y_list ]          
			  if len(occurrences_x)!=0 or len(occurrences_y)!=0:
				  print('error: found a common element in the validation set and in the training set. Please re-run the script with the --re-trainingset option'); sys.exit()
              
			  
          return  [X_train, Y_train,X_train_files,Y_train_files, X_validate, Y_validate, X_validate_file, Y_validate_file,TRAIN_PATH_x_list,TRAIN_PATH_y_list, VALID_PATH_x_list, VALID_PATH_y_list]

#%%
########################################################################################################################
################################################       MAIN     ########################################################
################################ ############################################################ ##########################

if __name__ == '__main__':

    #%%      
      #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      #~#~#~#~#~#~#~#~#~#~#~#~#            Input Parsing        #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

      train_flag,test_flag,predict_flag,PATH_x,PATH_y,model_dir,gpu_num,renovate_trainingset,renovate_testset,btch_s,trnngset_div,checkpoint_dir,checkpoint_basename,checkpoint_step, cores,output,num_labels,keep_path_list,fltr1stnmb,dims=input_parsing(sys.argv[1:])

      print('training: ' +str(train_flag))
      print('test:'+str(test_flag))
      print('predict:'+str(predict_flag))
      print('images path:'+PATH_x)  
      print('dimensions:'+str(dims))
      print('filter:'+str(fltr1stnmb))
      if PATH_y is not None:
		print('labels path:'+PATH_y)
      if model_dir is not None:
		print('model dir:'+model_dir)
      print('gpu device:'+gpu_num)
      print('renovate traningset:'+str(renovate_trainingset))
      print('traningset division:'+str(trnngset_div))
      print('checkpoint dir:'+str(checkpoint_dir))
      print('checkpoint basename:'+str(checkpoint_basename))      
      print('checkpoint step:'+str(checkpoint_step))  
      print('number of threads:'+str(cores))
      
      #Set Gpu device
      os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

      IMG_WIDTH = dims[0]
      IMG_HEIGHT = dims[1]
      IMG_LENGTH = dims[2]  
      fbname=str(IMG_WIDTH)+'x'+str(IMG_HEIGHT)+'x'+str(IMG_LENGTH)
      ext_name_train="UNETBrainSeg"
      ext_name_test="UNETBrainSeg"
      metrics_bname=ext_name_train+"_"+fbname+"_"+type_str+"_"+"filter"+str(fltr1stnmb)+"_btchs"+str(btch_s)+"_iter"+str(iterations)+"_dp"+str(drop_rate)+"_"+loss_type

      
      
      if test_flag:
        if PATH_x is None or PATH_y is None:
                     sys.exit()
        else:
            TEST_PATH_x = PATH_x
            TEST_PATH_y = PATH_y
      elif train_flag:
         if PATH_x is None or PATH_y is None:
                 sys.exit()
         else:
             TRAIN_PATH_x = PATH_x
             TRAIN_PATH_y = PATH_y
      elif predict_flag:
         if PATH_x is None :
                 sys.exit()
         else:
             Predict_PATH_x = PATH_x 
    
   
      if model_dir is None:
         model_dir=os.getcwd()+'/outputdir'
      trymakedir(model_dir)

      if checkpoint_dir is None:
        checkpoints_dir =   model_dir + "/checkpoints/"
        trymakedir(checkpoints_dir) 
        checkpoint_dir = checkpoints_dir + ext_name_train+"_"+fbname+"_"+"filter"+str(fltr1stnmb)+"_dp"+str(drop_rate)+"_"+loss_type+"/"
        trymakedir(checkpoint_dir) 
      if checkpoint_basename is None:
        checkpoint_basename = ext_name_train
        
      save_dir=model_dir+"/saved/"
      trymakedir(save_dir)
      graphs_dir=model_dir + "/graphs"
      towritedir=graphs_dir+ext_name_train+"_"+fbname+"/"
    
         
#%%      
      #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      #~#~#~#~#~#~#~#~#~#~#~#~# Get Data and preprocessing it  #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#_
      #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      
 
######### Trainingset
       
      if train_flag: 
          loss_list_file=save_dir+"loss"+metrics_bname+".npy"
          meanloss_list_file=save_dir+"meanloss"+metrics_bname+".npy"          
          
          trymakedir(graphs_dir)
          trymakedir(towritedir)

          train_i=int(np.random.uniform(0,trnngset_div))   

          #TRAINING SAVE FILES

          ext_image_filename=ext_name_train+"_"+fbname+"_"+image_type_str
          ext_label_filename=ext_name_train+"_"+fbname+"_"+label_type_str
          
          training_vars=get_trainingset(TRAIN_PATH_x,TRAIN_PATH_y,renovate_trainingset,keep_path_list,trnngset_div,train_i,validate_flag,ext_image_filename,ext_label_filename,dims,cores)

          X_train=training_vars[0]
          Y_train=training_vars[1]
          X_train_files=training_vars[2]
          Y_train_files=training_vars[3]
          X_validate=training_vars[4]
          Y_validate=training_vars[5]
          X_validate_file=training_vars[6]
          Y_validate_file=training_vars[7]
          TRAIN_PATH_x_list=training_vars[8]
          TRAIN_PATH_y_list=training_vars[9]
          VALID_PATH_x_list=training_vars[10]
          VALID_PATH_y_list=training_vars[11]

          loss_list_file=save_dir+"loss"+metrics_bname+".npy"
          meanloss_list_file=save_dir+"meanloss"+metrics_bname+".npy"          
          acc_list_file=save_dir+"accuracy"+metrics_bname+".npy"
          
          
          if not ( isfile(acc_list_file) and isfile(loss_list_file)  and isfile(meanloss_list_file) ):
                        renovate_stats=True
######### Testset
                
      if test_flag:
          
          #RENOVATE OR RESTORE TESTSET
          
          X_test_file=save_dir+"X_test"+"_"+ext_name_test+"_"+fbname+"_"+image_type_str+".npy"
          Y_test_file=save_dir+"Y_test"+"_"+ext_name_test+"_"+fbname+"_"+label_type_str+".npy"

          if not ( isfile(X_test_file) and isfile(Y_test_file) ):
                    renovate_testset=True
          
          if renovate_testset:
                X_test, Y_test = get_data(TEST_PATH_x,TEST_PATH_y, iamge_type,label_type,dims,ncores=cores)
                if 'X_train' in locals():
                    X_test = StandData(X_test, X_train)
                Y_test=Y_test.astype(label_type)
                print('save current testset...')
                print('saving current images of testset as'+X_test_file)
                np.save(X_test_file,X_test)
                print('saving current labels of testset as'+Y_test_file)
                np.save(Y_test_file,Y_test)
          else:
                print("restore last testset...")
                print('loading stored test images from: '+X_test_file)
                X_test=loadArray(X_test_file)
                print('loading stored test labels from: '+Y_test_file)
                Y_test=loadArray(Y_test_file)


          loss_list_file=save_dir+"loss"+metrics_bname+".npy"
          meanloss_list_file=save_dir+"meanloss"+metrics_bname+".npy"          
          acc_list_file=save_dir+"accuracy"+metrics_bname+".npy"
                
          if not ( isfile(acc_list_file) and isfile(loss_list_file) and isfile(Y_test_file) and isfile(meanloss_list_file) ):
                        renovate_stats=True
      if test_flag or validate_flag:
          #RENOVATE OR RESTORE METRICS  
          if compute_metrics_flag:   
              
              TPR_list_file=save_dir+"TPR"+metrics_bname+".npy"
              FPR_list_file=save_dir+"FPR"+metrics_bname+".npy"
              Precision_list_file=save_dir+"Precision"+metrics_bname+".npy"
              F1_score_list_file=save_dir+"F1_score"+metrics_bname+".npy"
              MCC_list_file=save_dir+"MCC"+metrics_bname+".npy"
              Youden_index_list_file=save_dir+"Youden_index"+metrics_bname+".npy"
              Dice_score_list_file=save_dir+"dice_score"+metrics_bname+".npy"
              Sensitivity_list_file=save_dir+"Sensitivity"+metrics_bname+".npy"
              Specificity_list_file=save_dir+"Specificity"+metrics_bname+".npy"
              Dice_score_list_file_all_file=save_dir+"dice_score"+metrics_bname+"_allImages.npy"
              Dice_score_train_list_file=save_dir+"dice_score_training"+metrics_bname+"_allImages.npy"

          if not ( isfile(Sensitivity_list_file) and isfile(Specificity_list_file) and isfile(Precision_list_file) and isfile(F1_score_list_file) and isfile(MCC_list_file) and isfile(Youden_index_list_file) and isfile(Dice_score_list_file) ):
                        renovate_metrics=True

              
          if renovate_metrics:
                                Precision_list=[]
                                F1_score_list=[]
                                MCC_list=[]
                                Youden_index_list=[]
                                Dice_score_list=[]
                                Sensitivity_list=[]
                                Specificity_list=[]
                                Dice_score_list_file_all=[]
          elif compute_metrics_flag and not renovate_metrics:
                                Precision_list=list(loadArray(Precision_list_file))
                                F1_score_list=list(loadArray(F1_score_list_file))
                                MCC_list=list(loadArray(MCC_list_file))
                                Youden_index_list=list(loadArray(Youden_index_list_file))
                                Dice_score_list=list(loadArray(Dice_score_list_file))
                                Sensitivity_list=list(loadArray(Sensitivity_list_file))
                                Specificity_list=list(loadArray(Specificity_list_file))
                                Dice_score_list_file_all=list(loadArray(Dice_score_list_file_all_file))

 
      if test_flag:
              Dice_score_list_file_all_test_file=save_dir+"dice_score_test"+metrics_bname+"_allImages.npy"
              if not isfile(Dice_score_list_file_all_test_file) or renovate_metrics:
				 
                                Dice_score_list_file_all_test=[]
              else:
                                Dice_score_list_file_all_test=list(loadArray(Dice_score_list_file_all_test_file))
             
              
              

                                
      if not isfile(Dice_score_train_list_file) or renovate_metrics:
				 
                                Dice_score_train_list=[]
      else:
                                Dice_score_train_list=list(loadArray(Dice_score_train_list_file))
      
######### Statistics
      if test_flag or train_flag:                               
		  #RENOVATE STATISTICS         
		  if renovate_stats:
				acc_list=[0]
				loss_list=[]
				meanloss_list=[]
		  
		  
		  else:
					  acc_list=list(loadArray(acc_list_file))
					  loss_list=list(loadArray(loss_list_file))
					  meanloss_list=list(loadArray(meanloss_list_file))

#%%
      #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      #~#~#~#~#~#~#~#~#~#~#~#~# initialize UNET ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
      if initialize_flag:
            if num_labels is None:
                        if 'Y_validate' in locals():
                                    num_labels=len(np.unique(Y_validate.astype(np.int32)))
                        elif 'Y_test' in locals():
                                    num_labels=len(np.unique(Y_test.astype(np.int32)))    
                        else:
                                    num_labels=default_num_labels
          
            if not train_flag:
              drop_rate=0.0

            unet = UNET_3D_light( loss_type=loss_type,
                                          drop_rate=drop_rate,
                                          use_softmax=use_softmax,
                                          filter1stnumb=fltr1stnmb,
                                          towritedir=towritedir,
                                          ckpt_dir=checkpoint_dir,
                                          ckpt_basename=checkpoint_basename,
                                          ckpt_step=checkpoint_step,
                                          ncores=cores,gpu_num=gpu_num,
                                          num_labels=num_labels)
      
      #%%
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#   Train  ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            

            if train_flag:
                  fail_list=[] 
                  Xt_numb=X_train.shape[0]
                  btch_s=btch_s
                  start_time = time.time()
                  #iterations=iterations+8
                  for i  in xrange(total_iter):
                        start_time_0 = time.time()
                        try:
		                unet.train(X_train, Y_train,iter=iterations,
		                           batch_count_thr=(Xt_numb/btch_s)-1,
		                           batch_size=btch_s,
		                           skip_iter=skp_iter,
		                           conv_kernel_size=conv_kernel_size,
		                           agument_batch=agument_flag)
                        except:
		                print('error while training on:' + X_train_file )
                        elapsed_time = time.time() - start_time_0
                        print("training time from iter "+str(i*iterations)+" to "+str((i+1)*iterations)+" : "+'%.3f' % elapsed_time+" sec")
                        
                        loss_list=loss_list+unet.loss_values
                        meanloss_list.append(np.mean(unet.loss_values))                        
                        np.save(loss_list_file,np.array(loss_list))
                        np.save(meanloss_list_file,np.array(meanloss_list))

                        ##### Validate
                        plt.close("all")


                        if validate_flag:
                              ix=range(X_validate.shape[0])
                              X_validate=NormImages(X_validate)
                              try:
                                    print('test on validationset')
                                    msk = unet.test(X_validate,
                                                    Y_validate,
                                                    indices=ix,
                                                    plot=False,
                                                    compute_accuracy=True,
                                                    compute_metrics=compute_metrics_flag,
                                                    plot_accuracy=False)
                                    acc_list.append(np.mean(unet.accuracy_avg))
                                    np.save(acc_list_file,np.array(acc_list))
                                    try:
                                        if compute_metrics_flag:
                                          
                                          Precision_list.append(np.mean(unet.precision_avg))
                                          F1_score_list.append(np.mean(unet.f1_score_avg))
                                          MCC_list.append(np.mean(unet.MCC_avg))
                                          Youden_index_list.append(np.mean(unet.Youden_index_avg))
                                          Dice_score_list.append(np.mean(unet.dice_score_avg))
                                          Sensitivity_list.append(np.mean(unet.sensitivity_avg))
                                          Specificity_list.append(np.mean(unet.specificity_avg))
                                          Dice_score_list_file_all.append(unet.dice_score)
                                    except:
                                        print('fail to retrive metrics')
																			
                                    try:
                                        np.save(Precision_list_file,np.array(Precision_list))
										
                                    except:
										print("fail to save file: "+Precision_list_file)
                                    try:
                                        np.save(F1_score_list_file,np.array(F1_score_list))
										
                                    except:
										print("fail to save file: "+F1_score_list)
                                    try:
                                        np.save(MCC_list_file,np.array(MCC_list))
										
                                    except:
										print("fail to save file: "+MCC_list_file)
                                    try:
                                        np.save(Youden_index_list_file,np.array(Youden_index_list))
										
                                    except:
										print("fail to save file: "+Youden_index_list_file)
                                    try:
                                        np.save(Dice_score_list_file,np.array(Dice_score_list))
										
                                    except:
										print("fail to save file: "+Dice_score_list_file)
                                    try:
                                        np.save(Sensitivity_list_file,np.array(Sensitivity_list))
										
                                    except:
										print("fail to save file: "+Sensitivity_list_file)
                                    try:
                                        np.save(Specificity_list_file,np.array(Specificity_list))
										
                                    except:
										print("fail to save file: "+Specificity_list_file)
                                    try:
                                        np.save(Dice_score_list_file_all_file,unet.dice_score)
										
                                    except:
										print("fail to save file: "+Dice_score_list_file_all_file)                                          
                                          
                              except:
                                   fail_list.append(ix)
                                   print("fail validation test on: idx "+str(ix))
                                    
                              
                              

                        ##### Restore Trainingset
                        if train_flag:    
                            
                            train_i_old=train_i
                            train_i = train_i_old + 1
                            if train_i >= trnngset_div:
                            	train_i=0
                            
                            X_train_file = X_train_files[train_i]
                            Y_train_file = Y_train_files[train_i]
                            
                            if not ( isfile(X_train_file) and isfile(Y_train_file) ):
                                                renovate_trainingset=True                             
                            
                            if renovate_trainingset:
                                                TRAIN_PATH_x,TRAIN_PATH_y = data_partition(TRAIN_PATH_x_list,TRAIN_PATH_y_list,train_i,trnngset_div)
                                                print("renovate the dataset...")
                                                print  (TRAIN_PATH_x[0]+" to "+TRAIN_PATH_x[-1])
                                                print  (TRAIN_PATH_y[0]+" to "+TRAIN_PATH_y[-1])
                                                unet.X_train = None
                                                unet.Y_train = None
                                                del X_train
                                                del Y_train 
                                                start_time_load = time.time()
                                                X_train, Y_train = get_data(TRAIN_PATH_x,TRAIN_PATH_y, iamge_type,label_type,dims,ncores=cores)
                                                X_train = StandData(X_train, X_train)
                                                Y_train=Y_train.astype(label_type)
                                                elapsed_time = time.time() - start_time_load
                                                print("time from loading data"+" : "+'%.3f' % elapsed_time+" sec")
                                                print('save current trainingset...')
                                                np.save(X_train_file,X_train)
                                                np.save(Y_train_file,Y_train)     
                            else:

                                                print("restore last trainingset...")
                                                unet.X_train = None
                                                unet.Y_train = None
                                                del X_train
                                                del Y_train 
                                                print('loading stored training images from: '+X_train_file) 
                                                X_train=loadArray(X_train_file)
                                                print('loading stored training labels from: '+X_train_file) 
                                                Y_train=loadArray(Y_train_file)               
                  
                  elapsed_time = time.time() - start_time         
                  print("Total training time after "+str(total_iter*iterations)+" : "+'%.3f' % elapsed_time+" sec")
                  print "Done"     
                  
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#  Test  ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            
            if not train_flag and test_flag:
                  plt.close("all")
                  ix=range(X_test.shape[0])
                  X_test=NormImages(X_test)
                  msk = unet.test(X_test,
                                  Y_test,
                                  indices=ix,
                                  plot=False,
                                  plot_separately=True,
                                  compute_accuracy=True,
                                  compute_metrics=compute_metrics_flag,
                                  plot_accuracy=False,
                                  print_separately=True,
                                  plot_ground_truth=False)
                  np.save(Dice_score_list_file_all_test_file,unet.dice_score)
                  plt.figure(),plt.plot(acc_list,color='b'),plt.title("Net Performace Metrics"),
                  plt.plot([0]+Sensitivity_list,color='m'),
                  plt.plot([0]+Specificity_list,color='c'),                  
                  plt.plot([0]+Precision_list,color='r'),
                  plt.plot([0]+Dice_score_list,color='k'),
                  plt.plot([0]+MCC_list,color='g'),
                  plt.plot([0]+Youden_index_list,color='y'),
                  plt.legend(["Accuracy","Sensitivity","Specificity","Precision","Dice_score","MCC","Youden index"])
                  plt.xlabel("Iterations")

                  plt.figure(),plt.plot(meanloss_list,color="orange"),plt.title("Mean Loss"),plt.xlabel("Iterations")
                  

            if predict_flag:
				
				if os.path.isdir(Predict_PATH_x):
					predict_dir=Predict_PATH_x
					predict_basename=os.path.basename(predict_dir)
					predict_dir=predict_dir.replace('//','/')
					if predict_dir[-1] == '/':
						predict_basename=os.path.basename(predict_dir[:-1])
					else:
						predict_basename=os.path.basename(predict_dir)
					if output is None:
						outout_predict_dir=os.path.dirname(Predict_PATH_x)
					else:
						if os.path.isdir(output):
							outout_predict_dir=output
						else:
							print('output must be a directory') 
							sys.exit()
					#outputdir=outout_predict_dir+'/'+predict_basename+'_predicted'
					outputdir=outout_predict_dir
					trymakedir(outputdir)
					images_dir=predict_dir  
					images_list=os.listdir(images_dir)
					images_list.sort()                         
					for inputfile in images_list:
						print(inputfile)
						T1_Struct=nib.load(images_dir+'/'+inputfile)
						input_idx=inputfile.find(".")
						outputfile=outputdir+'/'+inputfile[0:input_idx]+'_predicted.nii.gz'
						T1_img = T1_Struct.get_data();
						T1_aff=T1_Struct.get_affine();
						T1_header=T1_Struct.get_header();
						img=NormData(np.squeeze(T1_img));
						img = np.expand_dims(skresize( img ,dims, mode='constant',order=0), axis=-1);
						img = np.expand_dims(img , axis=0);
						seg_T1=unet.predict(img,plot=False)
						seg_T1=skresize( seg_T1 , T1_img.shape, mode='constant',order=0);
						seg_T1_int_Struct = nib.Nifti1Image(seg_T1, affine=T1_aff, header=T1_header);
						seg_T1_int_Struct.to_filename(outputfile) 
                        
				else: 						
						#load T1
						T1_img,T1_header,T1_aff=load_nib(Predict_PATH_x)
						img=tf_resp(T1_img,dims)
						
						#Predict segmentation    
						predictedSeg=unet.predict(img);
						predictedSeg=skresize( predictedSeg , T1_img.shape, mode='constant',order=0);
						
						#save results
						seg_T1_int_Struct = nib.Nifti1Image(integerize_seg(predictedSeg), affine=T1_aff, header=T1_header);
						if output is None:
							outout_predict_dir=os.path.dirname(Predict_PATH_x)
							predict_basename=os.path.basename(Predict_PATH_x)
							predict_basename=predict_basename[:predict_basename.find('.')]
							output_file=outout_predict_dir+'/'+predict_basename+'_segmentation.nii.gz'
						else:
							if os.path.isdir(output):
								outout_predict_dir=output
								predict_basename=os.path.basename(Predict_PATH_x)
								predict_basename=predict_basename[:predict_basename.find('.')]
								output_file=outout_predict_dir+'/'+predict_basename+'_segmentation.nii.gz'		
							else:
								output_file=output					
								
						seg_T1_int_Struct.to_filename(output_file)   
