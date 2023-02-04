#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1F8Lz6l4iPhv0GiO_zZ1L0x1Hfc5iU-Do -O checkpoint.zip
    unzip checkpoint.zip
fi
