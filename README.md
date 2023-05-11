# Brain Tumor Detection CNN

## Problem Statement 
Can a Convolutional Nueral Network classify MRI brain scans as healthy or having a tumor present better than a baseline score?

## Data Processing and Source
The data consists of over 3000 images sourced from:https://www.kaggle.com/abhranta/brain-tumor-detection-mri
Both classes were balanced.
The Yes folder represents tumor present scans.
The No folder represents healthy brain scans.
The Pred folder is unseen data used for further testing.
These files were visually inspected and then preprocessed for use in the CNN by cropping them to be homogenius and broken into matrices.

## Modeling 
A CNN was used with relu activation and sigmoid output layers. Epochs and batch sizes were tweaked. 
A callback was coded to ensure early stopping when the validation loss stagnated. 

## Results
With a best validation accuracy of 98.6%, this model outperformed a baseline accuracy of 50%

## Deployement 
The model was saved and exported to a simple streamlit app. This app can be used to classify unseen data. 

## Recommendations
This model could be further improved with a higher quantity of data as well as higher quality data and processing.
