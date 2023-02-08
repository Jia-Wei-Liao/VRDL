#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1FLP4PS8i5SZEHY4oRLBybqTqfpBGMoTW -O dataset.zip
    unzip dataset.zip
fi
