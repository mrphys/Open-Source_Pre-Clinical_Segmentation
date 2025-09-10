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
