#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1Eof7YM9Dt9WxNngZIymn_nDYhb5g3YN0 -O checkpoint.zip
    unzip checkpoint.zip
fi
