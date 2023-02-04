#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1F5T59FBI5n8YddrVVcI-dlOzOE33wXVx -O data.zip
    unzip data.zip
fi
