FROM ubuntu:latest

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh \
    && bash ./Anaconda3-2019.10-Linux-x86_64.sh -b -p $HOME/anaconda3\
    && export PATH=""

