#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1EjjSCNMrTS55jWBEVbsMzqJsLOGZHFd7 -O checkpoint.zip
    unzip checkpoint.zip
fi
