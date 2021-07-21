#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

#######################################################################################################################
#######################################################################################################################
###################		            							    ###################
###################	title:	            	                            	                    ###################
###################		           							    ###################
###################	description:	                                                            ###################
###################	version:	    0.1.0     				                    ###################
###################	notes:	        need to Install:  py modules - matplotlib,nibabel,skimage,  ###################
###################								vtk                 ###################
###################	bash version:   tested on GNU bash, version 4.3.48			    ###################
###################		           							    ###################
###################	autor: gamorosino     							    ###################
###################		           							    ###################
#######################################################################################################################
#######################################################################################################################
####################                                                                            #######################
####################                             update: added plot_overlay                     #######################
####################                                                                            #######################
#######################################################################################################################
#######################################################################################################################

Created on Thu Nov 22 11:06:06 2018

@author: gamorosino
"""
      ########################################################################################################################
      #########################################       Import Modules       ###################################################
      ########################################################################################################################
import sys
import Tkinter as tk
import matplotlib
import nibabel as nib
import ntpath
import numpy as np
import os
import tkFileDialog
from matplotlib import pyplot  as plt, gridspec
from matplotlib.path import Path
from matplotlib.widgets import Button, Cursor
from matplotlib.widgets import LassoSelector
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from externalClasses import VertSlider
from matplotlib.widgets import MultiCursor
      
from plot_track import get_track_vects
      
plt.close('all')
      
def OrtoView(Array3D,Mask=None,colormap='gray', alpha=0.4, Roispacing=None, head_nc=None, affine_nc=np.eye(4)):      
      ########################################################################################################################
      #######################################       Declare Globals      #####################################################
      ########################################################################################################################
      
      global K,I,J
      
      ### Slice class ###
      class Slice(object):
          def __init__(self,position):
              self.pos = position
      
      ###  Mesh class ###
      class Mesh(object):
          def __init__(self, visibility):
              self.vis = visibility
      ### ROI class ###
      class ROI(object):
          saved=""
          def __init__(self, Arraydata):
              self.data = Arraydata
      
      class ArrayNiiClass(object):
          fname = ""
          def __init__(self, axial):
              self.axial = axial
      
      class Tracking(object):
      
          trackname = ""
          output = ""
          error  = ""
          def __init__(self, trackname):
              self.trackname = trackname
      
          def plot(self):
              track_path = self.trackname
              if os.path.isfile(track_path):
                  plot_track(track_path,0.5,axarr[1,0])
                  mesh.vis=1
              else:
                  print "Error: not a valid track file to plot in the filed trackname"
      
          def click_plot(self,event):
              self.plot()
      
      ########################################################################################################################
      #####################################        Functions Declaration      ################################################
      ########################################################################################################################
      
      
      # update 3d mesh
      
      def mesh_update(Array,spacing):
      
          print 'Update mesh'
          try:
              clear_3Daxis()
              axarr[1,0].set_visible(mesh.vis)
              verts, faces = measure.marching_cubes(Array, 0, spacing=spacing)
              axarr[1, 0].plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                   lw=0.2, edgecolor="red", color="red",
                                   alpha=0.5)
              plt.draw()
              plt.show()
          except IndexError:
              pass
      
      def clear_3Daxis():
      
          axarr[1,0].clear()
          axarr[1,0].axis("off")
      
      # update ROI to show
      
      def update_roi(slice,asse):
      
          grid2show = np.copy(1 * slice)
          grid2show = (1 * (grid2show == 0)) * np.nan
          grid2show[np.nonzero(slice)] = 1
          masked_array = np.ma.array(grid2show, mask=np.isnan(grid2show))
          cmap = matplotlib.cm.autumn
          cmap.set_bad('red', 0.0)
          asse.imshow(masked_array, interpolation='nearest', cmap=cmap, alpha=alpha)
          plt.draw()
      
      def reset_axis(ax):
      
          Array3D = ArrayNiiClass.axial
          if ax==0:
              axarr[0, 0].clear();
              axarr[0, 0].imshow(Array3D[:, :, int(axialslice.pos)], cmap=colormap)
              axarr[0, 0].axis("off")
              axarr[0, 0].set_title('Axial')
          elif ax==1:
              axarr[0, 1].clear();
              axarr[0, 1].imshow(Array3D[:, int(sagittalslice.pos), :].transpose(1, 0), cmap=colormap)
              axarr[0, 1].axis("off")
              axarr[0, 1].set_title('Sagittal')
          elif ax==2:
              axarr[1, 1].clear();
              axarr[1, 1].imshow(Array3D[int(coronalslice.pos), :, :].transpose(1, 0), cmap=colormap)
              axarr[1, 1].axis("off")
              axarr[1, 1].set_title('Coronal')
          elif ax=='all':
              clear_axis()
              axarr[0, 0].imshow(Array3D[:, :, int(axialslice.pos)], cmap=colormap)
              axarr[0, 0].axis("off")
              axarr[0, 0].set_title('Axial')
              axarr[0, 1].imshow(Array3D[:, int(sagittalslice.pos), :].transpose(1, 0), cmap=colormap)
              axarr[0, 1].axis("off")
              axarr[0, 1].set_title('Sagittal')
              axarr[1, 1].imshow(Array3D[int(coronalslice.pos), :, :].transpose(1, 0), cmap=colormap)
              axarr[1, 1].axis("off")
              axarr[1, 1].set_title('Coronal')
          init_cursors(curs1, curs2, curs3)
      
      def setup_axis(img_axial, img_sagital,img_coronal ):
      
          clear_axis()
          axarr[0, 0].imshow(img_axial, cmap=colormap)
          axarr[0, 0].axis("off")
          axarr[0, 0].set_title('Axial')
          axarr[0, 1].imshow(img_sagital, cmap=colormap)
          axarr[0, 1].axis("off")
          axarr[0, 1].set_title('Sagittal')
          axarr[1, 1].imshow(img_coronal, cmap=colormap)
          axarr[1, 1].axis("off")
          axarr[1, 1].set_title('Coronal')
      
      def clear_axis():
      
          axarr[0, 0].clear();
          axarr[0, 1].clear();
          axarr[1, 1].clear();
      
      def clear_roi():
      
          print 'Clear ROI'
          roi.data = np.zeros_like(roi.data, dtype=int)
          reset_axis('all')
      
          mesh_update(roi.data, spacing)
      
      # save ROI as standard fullpath
      
      def roi_fastsave():
      
          nii_ROI = nib.Nifti1Image(roi.data[-1::-1,-1: :-1, -1::-1].transpose(1, 0, 2), affine=affine_nc, header=head_nc)
          filename, file_extension = os.path.splitext(ArrayNiiClass.fname)
          if file_extension == ".gz":
              filename = filename[0:filename.rfind('.')]
          filename_ROI = filename + '_ROI' + ".nii.gz"
          nii_ROI.to_filename(filename_ROI)
          print "ROI saved as " + filename_ROI
          roi.saved = filename_ROI
      
      # init_cursors
      
      def init_cursors(curs1,curs2,curs3):
      
          curs1.__init__(axarr[0, 0], color='r', lw=0.3)
          curs2.__init__(axarr[0, 1], color='r', lw=0.3)
          curs3.__init__(axarr[1, 1], color='r', lw=0.3)
      
      ### define cursors events ###
      def onClick(event):
          print 'pressed'
          curs1.set_active(0)
          curs2.set_active(0)
          curs3.set_active(0)
      
      
      # event on mouse relase
      
      def onRelease(event):
          print 'release'
          curs1.set_active(1)
          curs2.set_active(1)
          curs3.set_active(1)
      
      def convert_track(track_path):
          filename, file_extension = os.path.splitext(track_path)
          vtk_track_path=filename+".vtk"
          os.system("tckconvert "+ track_path + " " +vtk_track_path +" -force")
          return vtk_track_path
      
      def plot_track(track_path,downs,ax):
      
          xx_l, yy_l, zz_l, rgb_l = get_track_vects(track_path, downs)
      
          print "plot track..."
      
          ax.scatter(xx_l, yy_l, zz_l, c=rgb_l, depthshade=False, lw=0.01, s=5, alpha=1, antialiased=True)
          ax.axis("off")
      
      def initialize_plot(img_axial,img_sagital,img_coronal):
      
          ff, axarr = plt.subplots(2, 2)
          axi = axarr[0, 0].imshow(img_axial, cmap=colormap)
          axarr[0, 0].set_title('Axial')
          axarr[0, 0].axis("off")
          sag = axarr[0, 1].imshow(img_sagital, cmap=colormap)
          axarr[0, 1].set_title('Sagittal')
          axarr[0, 1].axis("off")
          cor = axarr[1, 1].imshow(img_coronal, cmap=colormap)
          axarr[1, 1].set_title('Coronal')
          axarr[1, 1].axis("off")
          plt.setp(axarr[1, 0].get_xticklabels(), visible=False)
          plt.setp(axarr[1, 0].get_yticklabels(), visible=False)
          
          I = int(coronalslice.pos)
          J= int(sagittalslice.pos)
          K = int(axialslice.pos)
          rois=roi.data[:, :, K]
          update_roi(rois,axarr[0, 0])
          rois = roi.data[:, J, :].transpose(1, 0)
          update_roi(rois,axarr[0, 1])
          rois = roi.data[I, :, :].transpose(1, 0)
          update_roi(rois,axarr[1, 1])
          
          #gs = gridspec.GridSpec(2, 1, width_ratios=[10, 13], height_ratios=[5, 7])
          #axarr[1, 0] = ff.add_subplot(gs[1], projection='3d', axis_bgcolor='black')
          axarr[1, 0].axis("off")
          return ff, axarr, axi ,sag, cor
      
      def reset_plot(ff, img_axial,img_sagital,img_coronal):
      
          axi = axarr[0, 0].imshow(img_axial, cmap=colormap)
          axarr[0, 0].set_title('Axial')
          axarr[0, 0].axis("off")
          sag = axarr[0, 1].imshow(img_sagital, cmap=colormap)
          axarr[0, 1].set_title('Sagittal')
          axarr[0, 1].axis("off")
          cor = axarr[1, 1].imshow(img_coronal, cmap=colormap)
          axarr[1, 1].set_title('Coronal')
          axarr[1, 1].axis("off")
          plt.setp(axarr[1, 0].get_xticklabels(), visible=False)
          plt.setp(axarr[1, 0].get_yticklabels(), visible=False)
          gs = gridspec.GridSpec(2, 1, width_ratios=[20, 23], height_ratios=[10, 13])
          axarr[1, 0] = ff.add_subplot(gs[1], projection='3d', axis_bgcolor='black')
          axarr[1, 0].axis("off")
          return ff, axarr, axi ,sag, cor
      
      #load NifTI image
      
      def load_nii(file_path_nc):
          print "loading Image..."
      
          image_nc = nib.load(file_path_nc)
          head_nc = image_nc.get_header()
          affine_nc = image_nc.get_affine()
          Array3D = image_nc.get_data()
      
          if len(Array3D.shape) == 4:
              Array3D=Array3D[:,:,:,0]
          elif len(Array3D.shape) <3:
              print "Wrong Dimension"
              exit()
          Array3D = Array3D[-1::-1,-1::-1,-1::-1].transpose(1,0,2)
          
          spacing=head_nc.get_zooms()
          if len(spacing)==4:
              spacing=head_nc.get_zooms()[0:-1]
      
          return  Array3D,spacing, head_nc, affine_nc
      
      
      # Initialize Classes
      
      def initialize_class(Array3D,Mask=None):
          
          if Mask is None:
                Mask = np.zeros(Array3D.shape,  dtype=np.int)
          roi = ROI(Mask)
          I = Array3D.shape[0] / 2
          J = Array3D.shape[1] / 2
          K = Array3D.shape[2] / 2
      
          axialslice = Slice(K)
          sagittalslice = Slice(J)
          coronalslice = Slice(I)
      
          img_axial = Array3D[:, :, K];
          img_sagital = Array3D[:, J, :].transpose(1, 0);
          img_coronal = Array3D[I, :, :].transpose(1, 0);
      
          mesh = Mesh(0)
      
          return roi,axialslice,sagittalslice,coronalslice,img_axial,img_sagital,img_coronal,mesh
      
      def select_file():
          root = tk.Tk()
          root.withdraw()
          file_path_nc = tkFileDialog.askopenfilename(initialdir="/home/gamorosino/",
                                                          filetypes=(("File Nifty", "*.nii*"), ("All Files", "*.*")),
                                                          title="Select Image")
          root.update()
          root.destroy()
          return file_path_nc
      
      ########################################################################################################################
      #####################################       Image Selector     #########################################################
      ########################################################################################################################
      
      
      ArrayNiiClass.fname=""
      track =  Tracking("")
      
      ########################################################################################################################
      #####################################        Classes Declaration      ##################################################
      ########################################################################################################################
      
      Array3D = Array3D[-1::-1,-1::-1,-1::-1].transpose(1,0,2) 
      ArrayNiiClass.axial=Array3D
      if Mask is not None:
            Mask = Mask[-1::-1,-1::-1,-1::-1].transpose(1,0,2)
      roi,axialslice,sagittalslice,coronalslice,img_axial,img_sagital,img_coronal,mesh=initialize_class(Array3D,Mask)
      
      
      ###### Class for callback button #####
      
      
      class Index_Slices(object):
      
          ind_axial = axialslice.pos
          ind_sagittal = sagittalslice.pos
          ind_coronal =coronalslice.pos
      
          projection=0
      
          def next(self, event):
      
      
              if self.projection == 0:
      
                  self.ind_axial += 1
                  Array3D = ArrayNiiClass.axial
                  axialslice.pos = self.ind_axial
                  saxial.set_val(self.ind_axial)
                  img_axial = Array3D[:, :, self.ind_axial];
                  immag = img_axial
                  roig = roi.data[:, :, self.ind_axial]
                  axxg = axarr[0, 0]
      
      
              elif self.projection == 1:
      
                  self.ind_sagittal += 1
                  Array3D = ArrayNiiClass.axial
                  sagittalslice.pos = self.ind_sagittal
                  ssagittal.set_val(self.ind_sagittal)
                  immag = Array3D[:, self.ind_sagittal, :].transpose(1, 0);
                  roig = roi.data[:, self.ind_sagittal, :].transpose(1, 0)
                  axxg = axarr[0, 1]
      
              elif self.projection == 2:
      
                  self.ind_coronal += 1
                  Array3D = ArrayNiiClass.axial
                  coronalslice.pos = self.ind_coronal
                  scoronal.set_val(self.ind_coronal)
                  immag = Array3D[self.ind_coronal, :, :].transpose(1, 0);
                  roig = roi.data[self.ind_coronal, :, :].transpose(1, 0)
                  axxg = axarr[1, 1]
      
              reset_axis(self.projection)
              axxg.imshow(immag, cmap=colormap)
              #axi.set_data(immag)
              update_roi(roig, axxg)
              plt.draw()
      
      
      
          def prev(self, event):
      
      
              if self.projection == 0:
      
                  self.ind_axial -= 1
                  Array3D = ArrayNiiClass.axial
                  axialslice.pos = self.ind_axial
                  saxial.set_val(self.ind_axial)
                  img_axial = Array3D[:, :, self.ind_axial];
                  immag = img_axial
                  roig = roi.data[:, :, self.ind_axial]
                  axxg = axarr[0, 0]
      
              elif self.projection == 1:
      
                  self.ind_sagittal -= 1
                  Array3D = ArrayNiiClass.axial
                  sagittalslice.pos = self.ind_sagittal
                  ssagittal.set_val(self.ind_sagittal)
                  immag = Array3D[:, self.ind_sagittal, :].transpose(1, 0);
                  roig = roi.data[:, self.ind_sagittal,:].transpose(1, 0);
                  axxg = axarr[0, 1]
      
              elif self.projection == 2:
      
                  self.ind_coronal -= 1
                  Array3D = ArrayNiiClass.axial
                  coronalslice.pos = self.ind_coronal
                  scoronal.set_val(self.ind_coronal)
                  immag = Array3D[self.ind_coronal, :, :].transpose(1, 0);
                  roig = roi.data[self.ind_coronal, :, :].transpose(1, 0)
                  axxg = axarr[1, 1]
      
              reset_axis(self.projection)
              axxg.imshow(immag, cmap=colormap)
              update_roi(roig, axxg)
              plt.draw()
      
      
      ###### Class for callback_tools button #####
      
      class Index_tools(object):
      
          visibility=mesh.vis
      
          def select_nii(self,event):
      
              file_path_nc = select_file()
              ArrayNiiClass.fname=file_path_nc
              Array3D, spacing, head_nc, affine_nc  = load_nii(file_path_nc)
              ArrayNiiClass.axial = Array3D
              Mask = np.zeros(Array3D.shape, int)
              roi.data = Mask
              reset_axis("all")
              axarr[1, 0].set_visible(0)
      
          def save_roi(self,event):
      
              roi_fastsave()
      
          def clear_roi(self, event):
      
              clear_roi()
      
      
          def update_mesh(self,event):
      
              mesh.vis = 1
              self.visibility = mesh.vis
              mesh_update(roi.data,spacing)
      
          def hide_mesh(self,event):
      
              self.visibility = mesh.vis
              if  self.visibility == 1 :
                  self.visibility=0
                  mesh.vis=0
                  print "Hide ROI mesh"
              else:
                  self.visibility=1
                  mesh.vis = 1
                  print "Show ROI mesh"
      
              axarr[1, 0].set_visible(self.visibility)
      
      
      
      
      ###### Classe for update_slider #####
      
      
      class Index_Sliders(object):
      
          projection=0
      
          def update_slider(self,val):
      
              if self.projection == 0:
      
                  K = saxial.val
                  K = int(K)
                  Array3D = ArrayNiiClass.axial
                  axialslice.pos = int(K)
                  callback_axial.ind_axial = int(axialslice.pos)
                  imgs = Array3D[:, :, K];
                  rois=roi.data[:, :, K]
                  axxs=axarr[0, 0]
      
              if self.projection == 1:
      
                  J = ssagittal.val
                  J = int(J)
                  Array3D = ArrayNiiClass.axial
                  sagittalslice.pos = int(J)
                  callback_sagittal.ind_sagittal = int(sagittalslice.pos)
                  imgs = Array3D[:, J, :].transpose(1, 0);
                  axxs = axarr[0, 1]
                  rois = roi.data[:, J, :].transpose(1, 0)
      
      
      
              if self.projection == 2:
      
                  I = scoronal.val
                  I = int(I)
                  Array3D = ArrayNiiClass.axial
                  coronalslice.pos = int(I)
                  callback_coronal.ind_coronal = int(coronalslice.pos)
                  imgs = Array3D[I, :, :].transpose(1, 0);
                  axxs = axarr[1, 1]
                  rois = roi.data[I, :, :].transpose(1, 0)
      
              reset_axis(self.projection)
              axxs.imshow(imgs, cmap=colormap)
              #axxs.set_data(imgs)
              update_roi(rois, axxs)
              plt.draw()
      
      ###### Classe for mouse event #####
      
      class Index_Mouse(object):
      
          ind_axial     =   axialslice.pos
          ind_sagittal  =   sagittalslice.pos
          ind_coronal   =   coronalslice.pos
      
          projection = -1
      
      
          # event on mouse scroll
      
          def onScroll (self,event):
      
      
                  if axarr[0, 0].in_axes(event) == True:
                      self.projection = 0
                  elif axarr[0, 1].in_axes(event) == True:
                      self.projection = 1
                  elif axarr[1, 1].in_axes(event) == True:
                      self.projection = 2
      
                  if event.button == 'up':
      
      
                      if self.projection == 0:
      
                              self.ind_axial = axialslice.pos
                              self.ind_axial += 1
                              Array3D = ArrayNiiClass.axial
                              axialslice.pos = self.ind_axial
                              saxial.set_val(self.ind_axial)
                              img_axial = Array3D[:, :, self.ind_axial];
                              immag = img_axial
                              roig = roi.data[:, :, self.ind_axial]
                              axxg = axarr[0, 0]
      
      
                      elif self.projection == 1:
      
                              self.ind_sagittal = sagittalslice.pos
                              self.ind_sagittal += 1
                              Array3D = ArrayNiiClass.axial
                              sagittalslice.pos = self.ind_sagittal
                              ssagittal.set_val(self.ind_sagittal)
                              immag = Array3D[:, self.ind_sagittal, :].transpose(1, 0);
                              roig = roi.data[:, self.ind_sagittal, :].transpose(1, 0)
                              axxg = axarr[0, 1]
      
                      elif self.projection == 2:
      
                              self.ind_coronal = coronalslice.pos
                              self.ind_coronal += 1
                              Array3D = ArrayNiiClass.axial
                              coronalslice.pos = self.ind_coronal
                              scoronal.set_val(self.ind_coronal)
                              immag = Array3D[self.ind_coronal, :, :].transpose(1, 0);
                              roig = roi.data[self.ind_coronal, :, :].transpose(1, 0)
                              axxg = axarr[1, 1]
      
                      reset_axis(self.projection)
                      axxg.imshow(immag, cmap=colormap)
                      # axi.set_data(immag)
                      update_roi(roig, axxg)
                      plt.draw()
      
                  elif event.button == 'down':
      
                      if self.projection == 0:
      
                          self.ind_axial = axialslice.pos
                          self.ind_axial -= 1
                          Array3D = ArrayNiiClass.axial
                          axialslice.pos = self.ind_axial
                          saxial.set_val(self.ind_axial)
                          img_axial = Array3D[:, :, self.ind_axial];
                          immag = img_axial
                          roig = roi.data[:, :, self.ind_axial]
                          axxg = axarr[0, 0]
      
                      elif self.projection == 1:
      
                          self.ind_sagittal = sagittalslice.pos
                          self.ind_sagittal -= 1
                          Array3D = ArrayNiiClass.axial
                          sagittalslice.pos = self.ind_sagittal
                          ssagittal.set_val(self.ind_sagittal)
                          immag = Array3D[:, self.ind_sagittal, :].transpose(1, 0);
                          roig = roi.data[:, self.ind_sagittal, :].transpose(1, 0);
                          axxg = axarr[0, 1]
      
                      elif self.projection == 2:
      
                          self.ind_coronal = coronalslice.pos
                          self.ind_coronal -= 1
                          Array3D = ArrayNiiClass.axial
                          coronalslice.pos = self.ind_coronal
                          scoronal.set_val(self.ind_coronal)
                          immag = Array3D[self.ind_coronal, :, :].transpose(1, 0);
                          roig = roi.data[self.ind_coronal, :, :].transpose(1, 0)
                          axxg = axarr[1, 1]
      
                      reset_axis(self.projection)
                      axxg.imshow(immag, cmap=colormap)
                      update_roi(roig, axxg)
                      plt.draw()
      
      
      
      ########################################################################################################################
      #####################################      Initialize Plot     #########################################################
      ########################################################################################################################
      
      # Four axes, returned as a 2-d array
      
      ff, axarr, axi ,sag, cor = initialize_plot(img_axial,img_sagital,img_coronal)
      
      ########################################################################################################################
      #####################################       Add sliders        #########################################################
      ########################################################################################################################
      
      axcolor = 'darkgray'
      axial_offset=0.06
      sxaxial=0.518-axial_offset
      syaxial=0.5+axial_offset
      sdimxaxial=0.02
      sdimyaxial=0.3
      
      sxsagittal=(0.518-0.018)*2-axial_offset
      sysagittal=0.5+axial_offset
      sdimxsagittal=0.02
      sdimysagittal=0.3
      
      sxcoronal=(0.518-0.018)*2-axial_offset
      sycoronal=2*axial_offset
      sdimxcoronal=0.02
      sdimycoronal=0.3
      
      axslider_axial = plt.axes([sxaxial, syaxial, sdimxaxial, sdimyaxial], axisbg=axcolor)
      saxial = VertSlider(axslider_axial, '', 0, Array3D.shape[2], valinit=axialslice.pos)
      idx_sliderax=Index_Sliders()
      idx_sliderax.projection=0
      saxial.on_changed(idx_sliderax.update_slider)
      
      axslider_sagittal = plt.axes([sxsagittal, sysagittal, sdimxsagittal, sdimysagittal], axisbg=axcolor)
      ssagittal = VertSlider(axslider_sagittal, '', 0, Array3D.shape[1], valinit=sagittalslice.pos)
      idx_slidersag=Index_Sliders()
      idx_slidersag.projection=1
      ssagittal.on_changed(idx_slidersag.update_slider)
      
      axslider_coronal = plt.axes([sxcoronal,sycoronal , sdimxcoronal, sdimycoronal], axisbg=axcolor)
      scoronal = VertSlider(axslider_coronal, '', 0, Array3D.shape[0], valinit=coronalslice.pos)
      idx_slidercor=Index_Sliders()
      idx_slidercor.projection=2
      scoronal.on_changed(idx_slidercor.update_slider)
      
      
      
      ########################################################################################################################
      #####################################        Add Buttons       #########################################################
      ########################################################################################################################
      
      
      bdimxaxial=sdimxaxial
      bdimyaxial=bdimxaxial+0.4*bdimxaxial
      bnextxaxial=sxaxial
      bnextyaxial=syaxial
      bprexaxial=sxaxial
      bpreyaxial=syaxial+sdimyaxial
      
      # Axial
      bdimxaxial=sdimxaxial
      bdimyaxial=bdimxaxial+0.4*bdimxaxial
      bnextxaxial=sxaxial
      bnextyaxial=syaxial
      bprexaxial=sxaxial
      bpreyaxial=syaxial+sdimyaxial
      callback_axial = Index_Slices()
      callback_axial.projection=0
      axprevaxial = plt.axes([bnextxaxial, bnextyaxial, bdimxaxial,bdimyaxial])
      axnextaxial = plt.axes([bprexaxial, bpreyaxial, bdimxaxial,bdimyaxial])
      bnextaxial = Button(axnextaxial, '+')
      bnextaxial.on_clicked(callback_axial.next)
      bprevaxial = Button(axprevaxial, '-')
      bprevaxial.on_clicked(callback_axial.prev)
      
      #Sagittal
      bdimxsagittal=sdimxsagittal
      bdimysagittal=bdimxsagittal+0.4*bdimxsagittal
      bnextxsagittal=sxsagittal
      bnextysagittal=sysagittal
      bprexsagittal=sxsagittal
      bpreysagittal=sysagittal+sdimysagittal
      callback_sagittal = Index_Slices()
      callback_sagittal.projection=1
      axprevsagittal = plt.axes([bnextxsagittal, bnextysagittal, bdimxsagittal,bdimysagittal])
      axnextsagittal = plt.axes([bprexsagittal, bpreysagittal, bdimxsagittal,bdimysagittal])
      bnextsagittal = Button(axnextsagittal, '+')
      bnextsagittal.on_clicked(callback_sagittal.next)
      bprevsagittal = Button(axprevsagittal, '-')
      bprevsagittal.on_clicked(callback_sagittal.prev)
      
      #Coronal
      bdimxcoronal=sdimxcoronal
      bdimycoronal=bdimxcoronal+0.4*bdimxcoronal
      bnextxcoronal=sxcoronal
      bnextycoronal=sycoronal
      bprexcoronal=sxcoronal
      bpreycoronal=sycoronal+sdimycoronal
      callback_coronal = Index_Slices()
      callback_coronal.projection=2
      axprevcoronal = plt.axes([bnextxcoronal, bnextycoronal, bdimxcoronal,bdimycoronal])
      axnextcoronal = plt.axes([bprexcoronal, bpreycoronal, bdimxcoronal,bdimycoronal])
      bnextcoronal = Button(axnextcoronal, '+')
      bnextcoronal.on_clicked(callback_coronal.next)
      bprevcoronal = Button(axprevcoronal, '-')
      bprevcoronal.on_clicked(callback_coronal.prev)
      
      
      ### ADD Button Clear ROI ###
      roibxoffset=0.02
      roibyoffset=0.01
      
      callback_tools = Index_tools()
      bxclearr=0.55
      byclearr=0.007
      bdimxclearr=0.11
      bdimyclearr=0.060
      axclearr = plt.axes([bxclearr, byclearr, bdimxclearr,bdimyclearr])
      bnclearr= Button(axclearr, 'Clear ROI')
      bnclearr.on_clicked(callback_tools.clear_roi)
      
   
      
      ### ADD Button track  ###
      
      bxtrack=0.125
      bytrack=0.007
      bdimxtrack=0.11
      bdimytrack=0.060
      axtrack = plt.axes([bxtrack, bytrack, bdimxtrack,bdimytrack])
      bntrack= Button(axtrack, 'load track')
      bntrack.on_clicked(track.click_plot)
      
      ### ADD Button Update Mesh ###
      callback_tools = Index_tools()
      bxupdate=bxtrack+roibxoffset+bdimxtrack
      byupdate=0.007
      bdimxupdate=0.09
      bdimyupdate=0.060
      axupdate = plt.axes([bxupdate, byupdate, bdimxupdate,bdimyupdate])
      bnupdate= Button(axupdate, 'Update')
      bnupdate.on_clicked(callback_tools.update_mesh)
      
      
      ### ADD Button Show/Hide Mesh ###
      callback_tools = Index_tools()
      bxresetm=bxupdate+roibxoffset+bdimxupdate
      byresetm=0.007
      bdimxresetm=0.11
      bdimyresetm=0.060
      axresetm = plt.axes([bxresetm, byresetm, bdimxresetm,bdimyresetm])
      bnresetm= Button(axresetm, 'Show/Hide')
      bnresetm.on_clicked(callback_tools.hide_mesh)
      
      ### ADD Button Load ###
      
      callback_tools = Index_tools()
      bxloadm=0+float(float(roibxoffset)/float(2))
      bdimxloadm=0.11
      bdimyloadm=0.060
      byloadm=1-bdimyloadm-roibyoffset
      axloadm = plt.axes([bxloadm, byloadm, bdimxloadm,bdimyloadm])
      bnloadm= Button(axloadm, 'Load File')
      bnloadm.on_clicked(callback_tools.select_nii)
      
      ########################################################################################################################
      #######################################     Define scroll event      ###################################################
      ########################################################################################################################
      
      indmous=Index_Mouse()
      indmous.projection=-1
      ff.canvas.mpl_connect('scroll_event',indmous.onScroll)
      
      ########################################################################################################################
      #######################################     Define Cursors     #########################################################
      ########################################################################################################################
      
      curs1 = Cursor(axarr[0, 0])
      curs2 = Cursor(axarr[0, 1])
      curs3 = Cursor(axarr[1, 1])
      
      curs1.connect_event('button_press_event',onClick)
      curs1.connect_event('button_release_event',onRelease)
      curs2.connect_event('button_press_event',onClick)
      curs2.connect_event('button_release_event',onRelease)
      curs3.connect_event('button_release_event',onRelease)
      curs3.connect_event('button_press_event',onClick)
      
      init_cursors(curs1,curs2,curs3)
      
      return axi,sag,cor

def plot_overlay(input_mask, alpha=0.5, **kwargs):
    colormap = kwargs.get('cmap', None)
    grid2show = np.copy(1 * input_mask)
    grid2show = (1 * (grid2show == 0)) * np.nan
    grid2show_one = np.copy(grid2show)
    grid2show_one[np.nonzero(input_mask)] = 1
    masked_array = np.ma.array(input_mask, mask=np.isnan(grid2show_one))
    if colormap == None:
        cmap = matplotlib.cm.autumn
    else:
        cmap = colormap
    cmap.set_bad('red', 0.0)
    plt.imshow(masked_array, interpolation='nearest', cmap=cmap, alpha=alpha)
    plt.draw()
