# PARCEL: Physics-based unsupervised contrastive representation learning for parallel MR imaging

## Abstract

With the successful application of deep learning in magnetic resonance imaging, parallel imaging techniques based on neural networks have attracted wide attentions. However, without high-quality fully sampled datasets for training, the performance of these methods tends to be limited. To address this issue, this paper proposes a physics based unsupervised contrastive representation learning (PARCEL) method to speed up parallel MR imaging. Specifically, PARCEL has three key ingredients to achieve direct deep learning from the undersampled k-space data. Namely, a parallel framework has been developed by learning two branches of model-based networks unrolled with the conjugate gradient algorithm; Augmented undersampled k-space data randomly drawn from the obtained k-space data are used to help the parallel network to capture the detailed information. A specially designed co-training loss is designed to guide the two networks to capture the inherent features and representations of the-to-be-reconstructed MR image. The proposed method has been evaluated on in vivo datasets and compared to five state-of-the-art methods, whose results show PARCEL is able to learn useful representations for more accurate MR reconstructions without the reliance on the fully-sampled datasets.
![architecture](https://user-images.githubusercontent.com/26486978/152712953-8f399d63-b57e-4f6c-bcb2-e042ef3d02dd.png)
  The pipeline of our proposed framework for physics-based unsupervised contrastive representation learning model
## How to use


## Clone repository


## dataset


## Requirements


## Training phase


## Test phase

## Results


## Acknowledgements
