# Set14 Super-Resolution


## Introduction of dataset
In this task, the dataset contains 291 training images with hight-resolution images and 14 test images with low resolution images. The test images is come from Set14 dataset. Since we don't have low-resolution of training data, we should generate low-resolution training images with 3 scale by bicubic interpolation at the first, and then split the dataset to training set and validation set. At the inference step, we should generate the hight-resolution images and upload to the CodaLab.


## Repository structure
      .
      ├──checkpoint
      |   └──SRFBN_28.3409
      |       ├──result
      |       ├──best_ckp.pth
      |       ├──options.json
      |       └──train_records.csv
      ├──data
      |   ├──__init__.py
      |   ├──common.py
      |   ├──LR_dataset.py
      |   └──LRHR_dataset.py
      ├──dataset
      |   ├──testing_lr_images
      |   └──training_hr_images
      ├──network
      |   ├──__init__.py
      |   ├──blocks.py
      |   ├──dbpn_arch.py
      |   ├──edsr_arch.py
      |   ├──rdn_arch.py
      |   └──srfbn_arch.py
      ├──options
      |   ├──options.py
      |   ├──test_EDSR.py
      |   ├──test_SRFBN.py
      |   ├──train_EDSR.py 
      |   └──train_SRFBN.py
      ├──solvers
      |   ├──__init__.py
      |   ├──base_solver.py
      |   └──SRSolver.py
      ├──utils
      |   └──util.py
      ├──inference.py
      ├──prepare_dataset.py
      ├──split_train_val.py
      ├──test.py
      └──train.py

Notice that `prepare_dataset.py`, `split_train_val.py`, and `inference.py` are programming by ourselves, the other is from https://github.com/Paper99/SRFBN_CVPR19


## Requirements
- python3
- skimage
- imageio
- pytorch
- tqdm
- pandas
- cv2


## Dataset
You can download the dataset on the Google Drive:  
https://drive.google.com/drive/folders/1akHIvXOUW0GieiRH0plqbrmQxXY_qp7R?usp=sharing


## Data pre-processing
Before training, you should prepare training set and validation set with hight resolution and low resolution. You can run the command as the following.
#### 1. Split the training and validation set
To split the dataset for training and validation, you can run this command:
```
python split_train_val.py --ratio <training number over validation number>
```

#### 2. Prepare training and validation set
To get the low resolution image for training, you can run this command:
```
python prepare_dataset.py --mode {train, val} --dataset {train, val}_HR_data
```


## Pre-trained weight
You can download the weight and checkpoint of our model and config on the Google Drive:  
https://drive.google.com/drive/folders/1eXMP_kCC5LtV5vtrxu1dqQBvYP0OVqUK?usp=sharing


## Training
To train the model, you can run this command:
```
python train.py -opt options/train_{SRFBN, EDSR}.json
```


## Inference
To inference the results, you can run this command:
```
python inference.py -opt options/test_{SRFBN, EDSR}.json
```
Notice that our best result is SRFBN.

## Reproducing submission
To reproduce our submission, please do the following steps:
1. [Getting the code](https://github.com/Jia-Wei-Liao/Set14_Dataset_Super-Resolution#Getting-the-code)
2. [Install the package](https://github.com/Jia-Wei-Liao/Set14_Dataset_Super-Resolution#requirements)
3. [Download the dataset](https://github.com/Jia-Wei-Liao/Set14_Dataset_Super-Resolution#dataset)
4. [Download the weight of model](https://github.com/Jia-Wei-Liao/Set14_Dataset_Super-Resolution#pre-trained-weight)
5. [Inference](https://github.com/Jia-Wei-Liao/Set14_Dataset_Super-Resolution#inference)


## Experiments
| method  | PSNR      |
| ------  | --------- |
| Bicubic | 26.0654   |
| EDSR+   | 28.0968   |
| SRFBN+  | 28.4085   |


## GitHub Acknowledgement
We thank the authors of these repositories:  
https://github.com/Paper99/SRFBN_CVPR19 


## Reference
[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPR, 2017.  
[2] Zhen Li, Jinglei Yang, Zheng Liu, Xiaomin Yang, Gwanggil Jeon, and Wei Wu, Feedback Network for Image Super-Resolution, CVPR, 2019.
