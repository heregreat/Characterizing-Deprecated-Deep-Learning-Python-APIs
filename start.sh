#!/bin/bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:10.0-runtime nvidia-smi
docker run --gpus all -it -v /home/nianliu:/root tensorflow/tensorflow:1.13.2-gpu bash
#pip3 install --user -r official/requirements.txt
#gsutil cp gs://dl-platform-public-nvidia/b191551132/restart_patch.sh /tmp/restart_patch.sh
