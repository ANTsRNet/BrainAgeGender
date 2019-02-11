# WIP

# App:  Age/Gender prediction

Deep learning app made for age/gender prediction using gray matter probability images using ANTsRNet

## Model training notes

* Training data: IXI, NKI, Kirby, and Oasis
* Unet model (see ``Scripts/Training/``).
* Template-based data augmentation
* Lower resolution training (template size = [136, 176, 176])

## Sample prediction usage

```
#
#  Usage:
#    Rscript doBrainAgeGenderPrediction.R grayMatterImage reorientationGrayMatterTemplate
#
#  MacBook Pro 2016 (no GPU)
#

$ Rscript Scripts/doBrainAgeGenderPrediction.R Data/Example/KKI2009-24-BrainSegmentationPosteriors2.nii.gz  Data/Template/S_template3_resampled2_GrayMatterProbability.nii.gz

Reading reorientation template Data/Template/S_template3_resampled2_GrayMatterProbability.nii.gz  (elapsed time: 0.1263192 seconds)
Using TensorFlow backend.
Loading weights file2019-02-11 13:12:14.377767: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
  (elapsed time: 4.208532 seconds)
Reading  Data/Example/KKI2009-24-BrainSegmentationPosteriors2.nii.gz  (elapsed time: 0.1675889 seconds)
Normalizing to template (developer note:  probably should switch to a quick rigid/affine registration  (elapsed time: 0.4719591 seconds)
Prediction and decoding (elapsed time: 14.32414 seconds)


********************************
Predicted age:  87.75776 years
Predicted gender:  -1.779992  ---> [0, 1] = [male, female]
********************************


Total elapsed time: 15.18434 seconds
```

