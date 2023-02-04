#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1FIgfW9wL8x0bqtiK6mD13DMbqhOBRdRP -O dataset.zip
    unzip dataset.zip
fi
