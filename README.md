# Selected Topics in Visual Recognition using Deep Learning

## Caltech-UCSD Birds-200-2011 Classification
In homework 1, we participate a bird classification competition on [Codalab](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07). This challenge provided 3,000 training images and 3,033 test images for the fine-grained classification. We rank 21st out of 100 participants at the end. The following table display our scores on the Codalab leaderboard.

| method                 | accuracy |
| ---------------------- | -------- |
| ResNet50               | 0.71283  |
| ResNet50 + ensemble    | 0.76525  |
| ResNeXt-101            | 0.79393  |
| ResNeXt-101 + ensemble | 0.81141  |

[[code](https://github.com/Jia-Wei-Liao/CUB_200_2011_Dataset_Classification)]
[report]

## Street View House Numbers Detection
In homework 2, we participate SVHN detection competition on [Codalab](https://competitions.codalab.org/competitions/35888?secret_key=7e3231e6-358b-4f06-a528-0e3c8f9e328e). This challenge provided 33,402 training images and 13,068 test images for digit detection. We rank 11st out of 83 participants at the end. The following table display our scores on the Codalab leaderboard.

| method      | mAP@0.5:0.95 | speed on P100 GPU (img/s) | speed on K80  GPU (img/s) |
| ----------- | ------------ | --------------------------| ------------------------- |
| Faster-RCNN | 0.389141     | 0.2                       | X                         |
| YOLOv4      | 0.419870     | 0.07364                   | 0.13696                   |

[[code](https://github.com/Jia-Wei-Liao/SVHN_Dataset_Detection)]
[report]

## Nuclei Instance Segmentation
In homework 3, we participate nuclei segmentation competition on [Codalab](https://competitions.codalab.org/competitions/35888?secret_key=7e3231e6-358b-4f06-a528-0e3c8f9e328e). This challenge provided 24 training images with 14,598 nuclei and 6 test images with 2,360 nuclei for instance segmentation. We rank 8st out of 84 participants at the end. The following table display our scores on the Codalab leaderboard.

| method       | backbone      | mAP       |
| ------------ | ------------- | --------- |
| Mask R-CNN   | ResNet-50-C4  | 0.244385  |
| Mask R-CNN   | ResNet-50-FPN | 0.240068  |
| Mask R-CNN   | ResNet-101-C4 | 0.242977  |
| Mask R-CNN   | ResNet-101-FPN| 0.241530  |

[[code](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5)]
[report]

## Set14 Super Resolution
In homework 4, we participate super-resolution competition on [Codalab](https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400). This challenge provided 291 training images with high-resolution images and 14 test images with low-resolution images for super-resolution. We rank 4st out of 85 participants at the end. The following table display our scores on the Codalab leaderboard.

| method  | PSNR      |
| ------  | --------- |
| Bicubic | 26.0654   |
| EDSR+   | 28.0968   |
| SRFBN+  | 28.4085   |

[[code](https://github.com/Jia-Wei-Liao/Set14_Dataset_Super-Resolution)]
[report]

## Ultrasound_Nerve_Segmentation
In the final project, we participate ultrasound nerve segmentation competition on [Kaggle](https://www.kaggle.com/c/ultrasound-nerve-segmentation). This challenge provided 5,635 training images and 5,508 test images for semantic segmentation. We rank 7st out of 924 participants at the end. The following table display our scores on Kaggle.

| method       | backbone        | private score |
| ------------ | --------------- | ------------- |
| UNet         | ResNet34        | 0.71031       |
| UNet         | ResNet50        | 0.70857       |
| UNet         | EﬀicientNet-b0  | 0.70233       |
| UNet         | EﬀicientNet-b1  | 0.72341       |

[[code](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation)]
[report]
