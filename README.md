# [Scale-equalizing Pyramid Convolution for Object Detection(CVPR2020)](https://arxiv.org/abs/2005.03101)
By Xinjiang Wang*, Shilong Zhang*, Zhuoran Yu, Litong Feng, Wei Zhang.

## Introduction

Feature pyramid has been an efficient method to extract features at different scales. Development over this method mainly focuses on aggregating contextual information at different levels while seldom touching the inter-level corre- lation in the feature pyramid. Early computer vision meth- ods extracted scale-invariant features by locating the fea- ture extrema in both spatial and scale dimension. Inspired by this, a convolution across the pyramid level is proposed in this study, which is termed pyramid convolution and is a modified 3-D convolution. Stacked pyramid convolutions directly extract 3-D (scale and spatial) features and outper- forms other meticulously designed feature fusion modules. Based on the viewpoint of 3-D convolution, an integrated batch normalization that collects statistics from the whole feature pyramid is naturally inserted after the pyramid con- volution. Furthermore, we also show that the naive pyramid convolution, together with the design of RetinaNet head, actually best applies for extracting features from a Gaus- sian pyramid, whose properties can hardly be satisfied by a feature pyramid. In order to alleviate this discrepancy, we build a scale-equalizing pyramid convolution (SEPC) that aligns the shared pyramid convolution kernel only at high- level feature maps. Being computationally efficient and compatible with the head design of most single-stage object detectors, the SEPC module brings significant performance improvement (> 4AP increase on MS-COCO2017 dataset) in state-of-the-art one-stage object detectors, and a light version of SEPC also has ∼ 3.5AP gain with only around 7% inference time increase. The pyramid convolution also functions well as a stand-alone module in two-stage object detectors and is able to improve the performance by ∼ 2AP.

## Get Started
You need to install mmdetection (version1.1.0 with mmcv 0.4.3) firstly.
All our self-defined modules are in ```sepc``` directory, and it has same folder organization as mmdetecion.
You can start your experiments with our modified train.py in ```sepc/tools``` or inference our model with test.py in ```sepc/tools```. 
More guidance can be found from [mmdeteion](https://github.com/open-mmlab/mmdetection).
## Models
The results on COCO 2017 val is shown in the below table.


| Method | Backbone | Add modules  | Lr schd | box AP | Download |
| :----: | :------: | :-------:  | :-----: | :----: | :------: |
| FreeAnchor | R-50-FPN | Pconv |  1x  | 39.7| [model](https://drive.google.com/open?id=1rwZeuT-VDGgfdt7IPidIOMh_guaIjxcO)  |
| FreeAnchor | R-50-FPN | IBN+Pconv | 1x  | 41.0| [model](https://drive.google.com/open?id=148IsbWcRTCDg8TpGDNt3P4McV4kekMRl) |
| FreeAnchor | R-50-FPN | SEPC-lite  | 1x  | 41.9| [model](https://drive.google.com/open?id=1Qb7WYtQVGrLDMnRwSXC-wnI9Vf3oLs4o) |
| FreeAnchor | R-50-FPN | SEPC | 1x      |  43.0| [model](https://drive.google.com/open?id=1rV7tZtjlfjdaNYeWKql1vhb1WPvjdWus) |
| Fsaf | R-50-FPN | baseline |  1x  | 36.8| [model](https://drive.google.com/open?id=1XlFlz8u0IqRw-HKy95VUNFrx3IKWM3q0)  |
| Fsaf | R-50-FPN | Pconv |  1x  | 38.6| [model](https://drive.google.com/open?id=1jtWmlPmZtFgZxD2QAZzVCAN6BnoAgzTF)  |
| Fsaf | R-50-FPN | IBN+Pconv | 1x  | 39.1| [model](https://drive.google.com/open?id=1d--0AjEdZEEbyP0JnFMmxyoIZUtpR9gj) |
| Fsaf | R-50-FPN | SEPC-lite  | 1x  | 40.5| [model](https://drive.google.com/open?id=1ODiNv14Bb44-RQrz-9cHtoNlbVKSItN6) |
| Fsaf | R-50-FPN | SEPC | 1x      |  41.6| [model](https://drive.google.com/open?id=1iLWrfXssXoGhZAQg9_vNeZfuwi_kXZu4) |
| Retinanet | R-50-FPN | Pconv |  1x  |37.0 | [model](https://drive.google.com/open?id=19cO_jxSZbbeyR5N_DWDbgM2wcyJEiD94)  |
| Retinanet | R-50-FPN | IBN+Pconv | 1x  | 37.8| [model](https://drive.google.com/open?id=12Mr2HG-0Qy_fkMDPw0eqtZ4t7G-M_pa6) |
| Retinanet | R-50-FPN | SEPC-lite  | 1x  | 38.5| [model](https://drive.google.com/open?id=1gq6bdWDZ83-5RzSeuUTpnOEeUyJvl2Zy) |
| Retinanet | R-50-FPN | SEPC | 1x      |  39.6| [model](https://drive.google.com/open?id=1HSG7CVFzHtv3McJ60nBXFw6pk91H-ONv) |


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
