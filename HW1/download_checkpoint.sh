#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1ESCx0g6_lh8nJ4AcZfx8nX2g0WIK1l5z -O checkpoint.zip
    unzip checkpoint.zip
fi
