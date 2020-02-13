FROM ubuntu:latest

RUN  apt-get update \
  && apt-get install -y wget gcc g++\
  && rm -rf /var/lib/apt/lists/* \
  && mkdir /root/object_reference_synthesis 

COPY . /root/object_reference_synthesis/

# Setup Anaconda 
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh \
    && bash ./Anaconda3-2019.10-Linux-x86_64.sh -b -p $HOME/anaconda3\
    && export PATH="$HOME/anaconda3/bin:$PATH" \
    && echo PATH="$HOME/anaconda3/bin:$PATH" >> ~/.bashrc \
    && conda env create -f /root/object_reference_synthesis/torch_geometric.yml
