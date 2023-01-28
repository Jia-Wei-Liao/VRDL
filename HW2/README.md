# SVHN Dataset Detection


In this homework, we implement the deep learning method to detect the digits images.First, we use detection package to construct Faster R-CNN model [3]. After the hyper-parameter tuning, we obtain a testing mAP of 0.389141. It’s not far from the baseline, but we can’t beat it. So, in order to beat the baseline, we implement the YOLOv4 model[4]. We refer to the code on the github and write the data processing, submission codeby ourselves. Finally, we exceed the baseline and get the test mAP of 0.41987.


##  Introduction of SVHN Dataset
The Street View House Numbers (SVHN) Dataset is the most widely-used dataset fordeveloping machine learning and object recognition. It’s obtained from house numbersin Google Street View images. In this homework, we contains 33,402 training imagesand 13,068 test images in SVHN dataset and need to train a model to detect digits. The model should predict fast and precise on inference time.


## Getting the code
You can download all the files in this repository by cloning this repository:
```
https://github.com/Jia-Wei-Liao/SVHN_Dataset_Detection.git
```


## File Structure
      .
      ├──FasterRCNN
      |   ├──checkpoint
      |   ├──detection (Reference [2])
      |   |   ├──coco_eval.py
      |   |   ├──coco_utils.py
      |   |   ├──engine.py
      |   |   ├──transforms.py
      |   |   └──utils.py
      |   |
      |   ├──src
      |   |   ├──dataset.py
      |   |   ├──model.py
      |   |   ├──transforms.py
      |   |   └──utils.py
      |   |
      |   ├──train
      |   |   ├──X.png (30000 pictures)
      |   |   └──digitStruct.mat
      |   |      
      |   ├──test
      |   |   └──X.png (13068 pictures)
      |   |      
      |   ├──00_mat2df.py (Reference [1])
      |   ├──01_train.py (Reference [3])
      |   ├──02_test.py
      |   ├──train_data.csv
      |   └──valid_data.csv
      |
      └──YOLOv4 (Reference [4])
          ├──cfg
          |   └──yolov4-pacsp.cfg
          |
          ├──data
          |   ├──hyp.scratch.yaml
          |   └──svhn.yaml (by myself)
          |
          ├──models
          |   ├──export.py
          |   └──models.py
          |
          ├──checkpoint
          |   └── demo_0.419pt (download from Goole Drive)
          |
          ├──train (download from Goole Drive)
          |   ├──X.png (30000 pictures)
          |   └──X.txt (30000 text file)
          |
          ├──valid (download from Goole Drive)
          |   ├──X.png (3402 pictures)
          |   └──X.txt (3402 text file)
          |
          ├──test (download from Goole Drive)
          |   └──X.png (13068 pictures)        
          |   
          ├──utils
          |   ├──activations.py
          |   ├──adabound.py           
          |   ├──autoanchor.py            
          |   ├──datasets.py          
          |   ├──evolve.sh            
          |   ├──gcp.sh           
          |   ├──general.py           
          |   ├──google_utils.py
          |   ├──layers.py
          |   ├──loss.py
          |   ├──metrics.py         
          |   ├──parse_config.py
          |   ├──metrics.py            
          |   ├──plots.py          
          |   ├──torch_utils.py           
          |   └──utils.py 
          |
          ├──generate_submission.py (by myself)
          ├──mat2yolo.py (by myself)
          ├──new_digitStruct.mat (we modify the mat file by MATLAB that scipy package can import)            
          ├──requirements.txt
          ├──split_train_valid.py (by myself)            
          ├──test.py 
          └──train.py


## Requirements
- `numpy == 1.17`
- `opencv-python >= 4.1`
- `torch == 1.6`
- `torchvision`
- `matplotlib`
- `pandas`
- `numpy`
- `scipy`
- `pycocotools`
- `tqdm`
- `pillow`
- `hdf5`
- `PIL`
- `tensorboard >= 1.14`


## Download
- You can download the dataset on the Google Drive:  
  - train: <https://drive.google.com/drive/folders/18jvQC966ovqfPn1nqW9YX1upIk8tJ6JF?usp=sharing>  
  - test: <https://drive.google.com/drive/folders/144QsIJxOH0mLTcXkvn4RvV2IQSwMkzFX?usp=sharing>
- You can download the weight on the Google Drive:  
<https://drive.google.com/drive/folders/1BPxTCnvXPHck3hg5QOFD1xJlMDZplKfh?usp=sharing>  


## Training
To train the model, you should move to folder YOLOv4 and run this command:
```
python train.py --data svhn.yaml --cfg cfg/yolov4-pacsp.cfg --weights checkpoint/yolov4.weights --device 0 --img 640 640 --batch-size 16
```


## Inference
To inference the results, you should move to folder YOLOv4 and run this command:
```
python generate_submission.py --data_path test --weight checkpoint/demo_0.419.pt
```


## Reproducing submission
To reproduce our submission, please do the following steps:
1. Getting the code
2. Install the package
3. Download the dataset and weight
4. Inference


## Results
Faster-RCNN and YOLOv4 achieve the following performance:
| Model                     | Faster-RCNN | YOLOv4   |
| ------------------------- | ----------- | ---------|
| test mAP@0.5:0.95         | 0.389141    | 0.41987  |
| speed on P100 GPU (img/s) | 0.2         | 0.07364  |
| speed on K80  GPU (img/s) | X           | 0.13696  |

You can open our Google Colab on this link:  
<https://github.com/Jia-Wei-Liao/SVHN_Dataset_Detection/blob/main/inference.ipynb>


## Reference
### Faster RCNN
[1] https://github.com/kayoyin/digit-detector/blob/master/construct_data.py  
[2] https://github.com/pytorch/vision/tree/main/references/detection  
[3] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

### YOLOv4
[4] https://github.com/WongKinYiu/PyTorch_YOLOv4
