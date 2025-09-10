## This GitHub repository is developed as part of the publication:
#
#  __Open-Source Pre-Clinical Image Segmentation: Mouse cardiac MRI datasets with a deep learning segmentation framework__
#
# submitted to the Journal of Cardiovascular Magnetic Resonance

# --------------------------------------------------------------

In this paper (a draft of which can be found in the 'Paper' folder here), we present the first publicly-available pre-clinical cardiac MRI dataset, along with an open-source DL segmentation model and a web-based interface for easy deployment (available at https://huggingface.co/spaces/mrphys/Pre-clinical_DL_segmentation).

**This GitHub repository contains the dataset and open-sourse model.  **

This dataset comprises of complete cine short-axis cardiac MRI images from 130 mice with diverse phenotypes.
It also contains expert manual segmentations of left ventricular (LV) blood pool and myocardium at end-diastole, end-systole, as well as additional timeframes with artefacts to improve robustness.

Using this resource, we developed an open-source DL segmentation model based on the UNet3+ architecture, with the training and inference code below.
These scripts are provided  an easy-to-follow tutorial

# --------------------------------------------------------------
## DATASET

The dataset is shared in HDF5 format, with one .h5 file per mouse (in the 'Data' folder. 
Each HDF5 file contains the full cine SAX dataset and the expert segmentations for the LV blood pool and myocardium. 
For each mouse the imaging data (h5 dataset name ‘Images’) is of size: matrix_size_x × matrix_size_y × number_of_slices × number_of_time_frames, where the matrix size is 256 × 256, the number of slices is in the range 8-11 and the number of timeframes is in the range 17-36. 
The segmentation data (h5 dataset name ‘Masks’) is the same size as the image data, with values in the range 0-2, representing voxels belonging to the background (0), the LV myocardium (1) and LV blood pool (2). 
A third parameter is given in the HDF5 file which identifies the cardiac timeframes which contain expert segmentation (h5 dataset name ‘Frames_with_ROI'). Cardiac timeframes which do not contain expert segmentation consist of only zeros in the ‘Masks’ data.

# --------------------------------------------------------------
## MODEL

In this study we developed an open-source DL segmentation model based on the UNet3+ architecture. 
The UNet3+ model consists of an advanced encoder-decoder network with deep supervision, designed to enhance multi-scale feature integration. 
The network features two layers per block, five scales, and batch normalization between layers. 
Final predicted labels are obtained by assigning each pixel to the class with the highest probability. 
