#!/bin/bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:10.0-runtime nvidia-smi
docker run --gpus all -it tensorflow/tensorflow:1.13.2-gpu bash
#pip3 install --user -r official/requirements.txt
