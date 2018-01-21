#!/usr/bin/env bash

IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')

nvidia-docker create -it --name Triplet_prediction --hostname DockerHost --mac-address 00:01:02:03:04:05 \
-v $(pwd):/Triplet_prediction -w=/Triplet_prediction \
-p 8888:8888 \
tensorflow/tensorflow:latest-gpu

docker start -i Triplet_prediction