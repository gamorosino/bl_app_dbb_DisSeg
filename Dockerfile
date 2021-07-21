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
      


