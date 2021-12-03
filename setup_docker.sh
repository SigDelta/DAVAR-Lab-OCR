#!/bin/bash

set -e

ROOT=$(cd $(dirname $0) && pwd )
echo $ROOT
PYTHON=${PYTHON:-"python3"}
PIP=${PIP:-"pip"}

# install the dependencies
# Requires torch>= 1.3.0 torchvision >= 0.4.1

# Dependencies from mmcv and mmdetection
$PIP install -r requirements.in

# Dependencies of DavarOCR
$PIP install mmpycocotools==12.0.3

$PIP install --use-deprecated=legacy-resolver warpctc-pytorch==0.2.1+torch16.cuda101

$PIP install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

###### install mmdetection #####
$PIP install mmdet==2.11.0

####### install davar-ocr ########
bash $ROOT/davarocr/setup_docker.sh

