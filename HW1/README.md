# Caltech-UCSD Birds-200-2011 Classification

In this homework, we implement the deep learning method to classify the birds images.
To avoid over-fitting, we use many augmentation to increase our training data. Then
we train the model with five fold cross validation. For the test time, we collect the
prediction which generate by five fold model and voting them.


## Environment
- `numpy == 1.18.5`
- `torch == 1.8.1 + cu101`
- `torchvision == 0.9.1 + cu101`


## File Structure
      .
      ├──data
         ├──training_images
            ├──XXXX.jpg
         ├──testing_images
            ├──XXXX.jpg
         ├──classes.txt            # contain 200 bird species and it's number
         ├──training_labels.txt    # filename and label mapping
         ├──testing_img_order.txt  # test filename
         ├──fold                   # we split training data into 5 fold
             ├──fold1.txt
             ├──fold2.txt
             ├──fold3.txt
             ├──fold4.txt
             ├──fold5.txt
      ├──src                       # functions inside
      ├──checkpoint                # trained model weights (download this file on Google Drive)
      ├──main01_training.py
      ├──main02_ensemble.py
      └──README.md


## Download Dataset
Run the commend:
```bash download_data.sh```


## Training
Open the `main01_training.py` and modify `params.file_root` by your file root.
- If you want to change the model, please set `params.model` to the `resnet50`, `resnet101`, `resnext50` or `resnext101`.
- If you want to do the k-fold cross validation, please set `params.K_fold == True`.


## Reproducing submission
There are two step that you need to notice.  

**Step 1.**
Run the commend to download the all pretrain model:
```bash download_checkpoint.sh```

**Step 2.** 
Open the `main02_ensemble.py` and modify `params.file_root` by your file root.
- You can adjust the checkpoint on the `ModelList` according to your requirement. 
- If you suffer from Out-Of-Memory errors, please reduce `params.batch_size`.
