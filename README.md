# Scale-equalizing Pyramid Convolution for object detection



By Xinjiang Wang*, Shilong Zhang*, Zhuoran Yu, Litong Feng, Wei Zhang.

## Introduction

Feature pyramid has been an efficient method to extract features at different scales. Development over this method mainly focuses on aggregating contextual information at different levels while seldom touching the inter-level corre- lation in the feature pyramid. Early computer vision meth- ods extracted scale-invariant features by locating the fea- ture extrema in both spatial and scale dimension. Inspired by this, a convolution across the pyramid level is proposed in this study, which is termed pyramid convolution and is a modified 3-D convolution. Stacked pyramid convolutions directly extract 3-D (scale and spatial) features and outper- forms other meticulously designed feature fusion modules. Based on the viewpoint of 3-D convolution, an integrated batch normalization that collects statistics from the whole feature pyramid is naturally inserted after the pyramid con- volution. Furthermore, we also show that the naive pyramid convolution, together with the design of RetinaNet head, actually best applies for extracting features from a Gaus- sian pyramid, whose properties can hardly be satisfied by a feature pyramid. In order to alleviate this discrepancy, we build a scale-equalizing pyramid convolution (SEPC) that aligns the shared pyramid convolution kernel only at high- level feature maps. Being computationally efficient and compatible with the head design of most single-stage object detectors, the SEPC module brings significant performance improvement (> 4AP increase on MS-COCO2017 dataset) in state-of-the-art one-stage object detectors, and a light version of SEPC also has ∼ 3.5AP gain with only around 7% inference time increase. The pyramid convolution also functions well as a stand-alone module in two-stage object detectors and is able to improve the performance by ∼ 2AP.

## Installation

## A quick demo


## Inference


## Models

## Training


## Citations
Please cite our paper in your publications if it helps your research:
```
@inproceedings{wang2020SEPC,
  title     =  {Scale-equalizing Pyramid Convolution for object detection},
  author    =  {Wang, Xinjiang and Zhang, Shilong and Yu, Zhuoran and Feng, Litong and Zhang, Wei},
  booktitle =  {CVPR},
  year      =  {2020}
}
```
