# PARCEL: Physics-based unsupervised contrastive representation learning for parallel MR imaging

## Abstract

With the successful application of deep learning to magnetic resonance (MR) imaging, parallel imaging techniques based on neural networks have attracted wide attention. However, in the absence of high-quality, fully sampled datasets for training, the performance of these methods is limited. To address this issue, this paper proposes a Physics-bAsed unsupeRvised Contrastive rEpresentation Learning (PARCEL) method to speed up parallel MR imaging. Specifically, PARCEL has a parallel framework to contrastively learn two branches of model-based unrolling networks directly from augmented undersampled k-space data. A sophisticated co-training loss with three essential components has been designed to guide the two networks in capturing the inherent features and representations for MR images. And the final MR image is reconstructed with the trained contrastive networks. PARCEL was evaluated on in vivo datasets and compared to five state-of-the-art methods. The results show that PARCEL is able to learn useful representations for more accurate MR reconstructions without relying on fully sampled datasets. 

![PARCEL](https://user-images.githubusercontent.com/26486978/155647766-d8e2bd4d-d456-439a-816a-69101a03ea4e.png)

  The pipeline of our proposed framework for physics-based unsupervised contrastive representation learning model
## How to use

This project is conducted on an Ubuntu 20.04 LTS (64-bit) operating system utilizing two NVIDIA RTX A6000 GPUs (each with a memory of 48GB). The following we will explain how to use this code to achieve PARCEL.
## Clone repository
```
git clone https://github.com/ternencewu123/PARCEL.git
```
## Training phase
Enter the path of the project and run the following scripts to train the parallel network.

```
python main.py -m='train' -trsr=0.2 -vsr=0.1
```
## Test phase
Enter the path of the project and run the following scripts to test the saved model.
```
python testdemo.py -tesr=1.0
```

## Results
![pd(2d)](https://user-images.githubusercontent.com/26486978/155674131-6cf302f2-1253-4045-94b1-90269c10a39e.png)


## Acknowledgements
[1].[fastMRI](https://fastmri.med.nyu.edu/).

[2].Aggarwal H K, Mani M P, Jacob M. MoDL: Model-based deep learning architecture for inverse problems. IEEE transactions on medical imaging, 2018, 38(2): 394-405.
