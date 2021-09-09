# SLC_ShiftingEnsemble

This repository contains the source code of the paper [Skin lesion classification by ensembles of deep convolutional networks and regularly spaced shifting](https://doi.org/10.1109/ACCESS.2021.3103410).

This code executes the **Shifted GoogLeNet+MobileNetV2** method for the HAM10000 dataset. The contents of this code are provided without any warranty. They are intended for evaluational purposes only.

<!-- ![Alt text](Example.PNG?raw=true "Operation method of SRCNN3D+RegSS") -->

### Pre-requisites

- Matlab (tested on v2020b or earlier). Deep learning toolbox is required to load GoogLeNet and MobileNetV2.
---

### Training

1. Open trainNets.m and set up the paths of the dataset
2. Run the script
---

### Testing

1. Open testNetGrids.m and set up the paths of the dataset
2. Run the script

### Evaluation

- computeStatsCV.m: computes the statistics of the 10-fols CV
- plotConfusionCV.m: computes the confusion matrices of the tested models
- plotModelsComparisonCV.m: plots the graph bar comparing all models

---

### Citation

Please, cite this work as:

K. Thurnhofer-Hemsi, E. López-Rubio, E. Domínguez and D. A. Elizondo, 
"Skin Lesion Classification by Ensembles of Deep Convolutional Networks and Regularly Spaced Shifting",
 in IEEE Access, vol. 9, pp. 112193-112205, 2021, doi: 10.1109/ACCESS.2021.3103410.
(https://ieeexplore.ieee.org/abstract/document/9508981)