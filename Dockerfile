FROM ubuntu:bionic-20200630 


MAINTAINER Gabriele

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install \
                    -y --no-install-recommends \
                     software-properties-common \
                     build-essential \
                     apt-transport-https \
                     ca-certificates \
                     gnupg \
                     software-properties-common \
                     wget \
                     ninja-build \
                      git \
                      zlib1g-dev \
                     apt-utils \
                     g++ \
                     libeigen3-dev \
                     libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev \
                     jq \
                     strace \
                     curl \
                     vim \
			zip \
			file
                     

## install Conda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-py27_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

## install Python Modules with Conda

RUN  conda install -c anaconda python=2.7 scikit-image=0.14.2 \
      && conda install -c anaconda python=2.7 requests=2.22.0 \
      && conda install -c conda-forge nibabel=2.5.1 \
      && conda install -c conda-forge pydicom=1.3.0 \
      && conda install -c conda-forge tqdm=4.38.0 \
      && conda install -c anaconda tensorflow-gpu=1.10.0 \
      && conda install -c conda-forge pyvista "vtk=8.1.2" "libnetcdf=4.6.2"
      

## Clone bl_app_dbb_DisSeg via Github

RUN cd / \
  && git clone git@github.com:gamorosino/bl_app_dbb_DisSeg.git \
  && cd bl_app_dbb_DisSeg \
  && git checkout docker

## Donwload UNET checkpoints and test data

RUN  cd / \
	&& /bin/bash -c 'source /bl_app_dbb_DisSeg/libraries/FILESlib.sh;\
	mkdir -p "/bl_app_dbb_DisSeg/scripts/checkpoints/";\
	checkpoints_dir=/bl_app_dbb_DisSeg/scripts/checkpoints/UNET_DATA_3D_T1_segment_8tms_0.2.9.2.6.1_256x256x256_filter12_dp0.15_sparse_softmax_cross_entropy_nosoftmax ; \
	mkdir -p ${checkpoints_dir}; \
	gdrive_download "https://drive.google.com/file/d/1P_Fk0eb2YEURXHkqp1Ek0WJab77K6aRH/view?usp=sharing"  ${checkpoints_dir}"/UNET_DATA_3D_T1_segment_8tms-66667.meta" ; \
	gdrive_download "https://drive.google.com/file/d/1fDgEzSUJp5DJbFUo4GNNGSuby7PLolWh/view?usp=sharing" ${checkpoints_dir}"/UNET_DATA_3D_T1_segment_8tms-66667.index" ; \
	gdrive_download "https://drive.google.com/file/d/1MqzZx6cS2JHKF9zV06odC8j8johl13ND/view?usp=sharing" ${checkpoints_dir}"/UNET_DATA_3D_T1_segment_8tms-66667.data-00000-of-00001" ; \ 
	echo "model_checkpoint_path: \"${checkpoints_dir}/UNET_DATA_3D_T1_segment_8tms-66667\" " >> ${checkpoints_dir}"/checkpoint" ; \
	echo "all_model_checkpoint_paths: \"${checkpoints_dir}/UNET_DATA_3D_T1_segment_8tms-66667\" " >> ${checkpoints_dir}"/checkpoint" ;  \


#make it work under singularity 
#https://wiki.ubuntu.com/DashAsBinSh 
RUN ldconfig

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

