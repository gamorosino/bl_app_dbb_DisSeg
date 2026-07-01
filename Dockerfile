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

## install Miniconda3 4.12.0 (pre-TOS enforcement)
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

## install Python Modules via pip
RUN pip install --no-cache-dir \
        "numpy<2" \
        scikit-image==0.19.3 \
        requests==2.28.0 \
        nibabel==4.0.2 \
        pydicom==2.3.1 \
        tqdm==4.64.0 \
        scipy \
        matplotlib \
        tensorflow-gpu==2.9.1 && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs

# Break Conda hardlinks that Singularity rootless cannot unpack
RUN find /opt/conda -type f -links +1 -exec bash -c '\
  for f; do \
    tmp="${f}.tmpcopy"; \
    cp -a --remove-destination "$f" "$tmp" && mv "$tmp" "$f"; \
  done' bash {} +

## Clone bl_app_dbb_DisSeg via Github
RUN cd / && git clone https://github.com/gamorosino/bl_app_dbb_DisSeg.git

#make it work under singularity
RUN ldconfig
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
