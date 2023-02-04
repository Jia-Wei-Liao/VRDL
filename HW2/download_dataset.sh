#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1FBPQCzUxY4C3tm8lCVXePKper9U1Jd0K -O dataset.zip
    unzip dataset.zip
fi
