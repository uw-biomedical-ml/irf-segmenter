FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

LABEL Description="irf-segmenter" Version="1.0.0"

ENV USER root
ENV TMPDIR /tmp

RUN apt-get update -y
RUN apt-get install -y python python-pip
RUN apt-get install -y python-tk
RUN apt-get install -y git
RUN apt-get install -y libopencv-dev python-opencv

RUN pip install --upgrade pip
RUN pip install h5py
RUN pip install pillow
RUN pip install matplotlib
RUN pip install keras==1.2.1
RUN pip install tensorflow-gpu==0.12.1

RUN git clone https://github.com/uw-biomedical-ml/irf-segmenter.git

RUN mkdir -p /data_in
RUN mkdir -p /data_out

WORKDIR /irf-segmenter
CMD python run.py $IN $OUT
