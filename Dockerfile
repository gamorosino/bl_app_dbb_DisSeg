FROM ubuntu:20.04

MAINTAINER Gabriele

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install \
                    -y --no-install-recommends \
                     software-properties-common \
                     build-essential \
                     apt-transport-https \
                     ca-certificates \
                     gnupg \
                     wget \
                     ninja-build \
                     git \
                     zlib1g-dev \
                     apt-utils \
                     g++ \
                     libeigen3-dev \
                     libgl1-mesa-dev \
                     libfftw3-dev \
                     libtiff5-dev \
                     jq \
                     strace \
                     curl \
                     vim \
                     zip \
                     file

## install Conda (Python 3)
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

## install Python Modules with Conda
RUN conda install -c anaconda python=3.9 && \
    conda install -c anaconda scikit-image=0.19.3 && \
    conda install -c anaconda requests=2.28.0 && \
    conda install -c conda-forge nibabel=4.0.2 && \
    conda install -c conda-forge pydicom=2.3.1 && \
    conda install -c conda-forge tqdm=4.64.0 && \
    conda install -c conda-forge tensorflow=2.9.1 cudatoolkit=11.2 cudnn=8.1.0 && \
    conda install -c conda-forge scipy matplotlib

## Clone bl_app_dbb_DisSeg via Github
RUN cd / && git clone https://github.com/gamorosino/bl_app_dbb_DisSeg.git

#make it work under singularity
RUN ldconfig
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
