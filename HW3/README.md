# Nuclei_Dataset_Segmentation


##  Introduction of nuclear dataset
The nuclei segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear.
Since we don't have annotation of data, we should generate annotation at the first,
and then use Detectron2[1] to register the custom datasets before training step.
At the inference step, we should generate the submission file and upload to the [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5#results).
The submission of format should follow by COCO results which include image\_id, category\_id, segmentation and score.
In addition, the segmentation result should be the RLE encoded format.


## Getting the code
You can download all the files in this repository by cloning this repository:
```
https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation.git
```


## Repository structure
      .
      ├──checkpoint
      |   └──best
      ├──dataset
      ├──src
      |   ├──coco_mask.py
      |   ├──trainer.py
      |   └──utils.py
      ├──inference.py 
      ├──make_annot.py
      └──train.py


## Requirements
```
tqdm
numpy
pandas
matplotlib
PIL
pillow
opencv-python
torch
torchvision
tensorboard
pycocotools
detectron2
```


## Dataset
#### 1. Download the dataset
You can download the dataset on the Google Drive:  
<https://drive.google.com/drive/folders/174K2gLqNqm2-m2MNHP4SZDbnx2L2JA4i?usp=sharing>
#### 2. Make training data annotations
To make the annotations, you can run this command:
```
python make_annot.py
```


## Pre-trained weight
You can download the weight and checkpoint of our model and config on the Google Drive:  
<https://drive.google.com/drive/folders/1rA5ajCevldGvTrzBxEYrSHnJP9Mx_Dsu?usp=sharing>

## Training
To train the model, you can run this command:
```
python train.py --model mask_rcnn_R_50_C4_1x --device cuda:0
```
- model: mask_rcnn\_R\_{50, 101}\_{C4, FPN}\_{1, 3}x
- device: cpu or cuda:{0, 1, 2, 3}


## Inference
To inference the results, you can run this command:
```
python inference.py --checkpoint 'best' --weight_name model_final.pth
```


## Reproducing submission
To reproduce our submission, please do the following steps:
1. [Getting the code](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#Getting-the-code)
2. [Install the package](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#requirements)
3. [Download the dataset](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#dataset)
4. [Download the weight of model](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#pre-trained-weight)
5. [Inference](https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation#inference)


## Experiments
| method       | backbone      | mAP       |
| ------------ | ------------- | --------- |
| Mask R-CNN   | ResNet-50-C4  | 0.244385  |
| Mask R-CNN   | ResNet-50-FPN | 0.240068  |
| Mask R-CNN   | ResNet-101-C4 | 0.242977  |
| Mask R-CNN   | ResNet-101-FPN| 0.241530  |


## Citation
If you find our work useful in your project, please cite:
```
@misc{
    title  = {nuclear_dataset_segmentation},
    author = {Jia-Wei Liao},
    url    = {https://github.com/Jia-Wei-Liao/Nuclear_Dataset_Segmentation},
    year   = {2021}
}
```


## Reference
[1] https://github.com/facebookresearch/detectron2  

[2] https://detectron2.readthedocs.io/en/latest/tutorials/install.html 
