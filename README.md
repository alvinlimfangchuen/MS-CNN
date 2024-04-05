# MS-CNN

Implementation of our published journal: [In-Air Hand Gesture Signature Recognition using Multi-Scale Convolutional Neural Networks](https://joiv.org/index.php/joiv/article/view/2359)

## Abstract

![MSCNN](https://alvinlfc.com/image/mscnn/mscnn.jpg)

The hand signature is a unique handwritten name or symbol serving as proof of identity. Its practicality and widespread use keep it prevalent in financial institutions for verifying and validating customer identities. However, the COVID-19 pandemic has underscored hygiene concerns with conventional touch-based hand signature recognition systems, which typically necessitate shared acquisition devices.

This paper introduces an in-air hand gesture signature recognition method employing convolutional neural networks (CNNs) to mitigate these concerns. We propose a shallow multi-scale CNN architecture utilizing kernel filters of sizes 3x3 and 5x5 to extract features at various scales parallely.

Our architecture was rigorously evaluated against other pre-trained models such as GoogleNet, AlexNet, VGG-16, and ResNet-50 using the In-Air Hand Gesture Database (iHGS) under same experimental settings. The results indicate that our proposed model surpasses competing architectures with a leading accuracy of 93.00%, while also being resource-efficient, averaging just 3 minutes and 33 seconds for training.

## Getting Started

To clone this repository and start exploring the MS-CNN project on your local machine.

```bash
git clone https://github.com/alvinlimfangchuen/MS-CNN.git
```

### Prerequisites

- MATLAB 2021a
- MATLAB Deep Learning Toolbox

### Dataset

The implementation of this project is based on the In-Air Hand Gesture Signature (iHGS) database, which is currently the only publicly available image-based dataset for in-air hand gesture signature recognition.
For more information on the iHGS database and to access it for your research, please visit the following link and contact the corresponding author:

[In-Air Hand Gesture Signature (iHGS) Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10439358/)

Please ensure you adhere to the dataset's usage guidelines and cite it appropriately in your work.

#### Sample of Dataset

![iHGS MHI Sample](https://alvinlfc.com/image/mscnn/sample.jpg)


## Citation

```bibtex
@article{lim2023inair,
  title={In-air Hand Gesture Signature Recognition using Multi-Scale Convolutional Neural Networks},
  author={Lim, A. F. C. and Khoh, W. H. and Pang, Y. H. and Yap, H. Y.},
  doi = {https://dx.doi.org/10.30630/joiv.7.3-2.2359},
  journal={International Journal on Informatics Visualization},
  volume={7},
  number={3-2},
  pages={2025--2031},
  year={2023}
}
```
