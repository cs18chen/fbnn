# fbnn


Pre-requisites:

The code is built at least with the following libraries: Python ;  Python Imaging Library ; Matplotlib ; Numpy.

Install cnn_gp package: https://github.com/cambridge-mlg/cnn-gp

Run Demo

For training deep model for each task, go to the related folder and follow the bellow steps:

Skin Lesion Segmentation

1-	Download the ISIC 2018 train dataset from https://challenge.isic-archive.com/data/  link and extract both training dataset and ground truth folders inside the dataset_isic18.

2-	Run Prepare_ISIC2018.py for data preperation and dividing data to train,validation and test sets.

3-	Run run_fvi_seg.py for training the proposed model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set.

4-	Run run_fvi_seg.py for test the proposed model using test set.

5-  For performance calculation and producing segmentation result, run evaluation.py. It will represent performance measures and will saves related figures and results in output folder.

References

https://github.com/happenwah/FVI_CV

https://github.com/zylye123/CARes-UNet

https://github.com/rezazad68/BCDU-Net

https://github.com/hula-ai/organ_segmentation_analysis

The baseline methods (Softmax Unet, Prob. Unet, Ensemble Unet, MC-Dropout, and SWAG ) were based on the implementation of the following references:

https://github.com/SteffenCzolbe/probabilistic_segmentation

https://github.com/wjmaddox/swa_gaussian








