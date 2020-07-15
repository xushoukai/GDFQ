# Generative-Low-bitwidth-Data-Free-Quantization

We provide PyTorch implementation for "Generative Low bitwidth Data Free Quantization".  

## Paper
* [Generative Low bitwidth Data Free Quantization](https://arxiv.org/abs/2003.03603) 
* Shoukai Xu<sup> *</sup>, Haokun Li<sup> *</sup>, Bohan Zhuang<sup> *</sup>, Jiezhang Cao, Jing Liu, Chuangrun Liang, Mingkui Tan<sup> *</sup> 
* The European Conference on Computer Vision(ECCV), 2020

<br/>

## Dependencies

* Python 3.6
* PyTorch 1.2.0
* dependencies in requirements.txt

<br/>

## Getting Started

### Installation

1. Clone this repo:

        git clone https://github.com/xushoukai/Generative-Low-bitwidth-Data-Free-Quantization.git
        cd Generative-Low-bitwidth-Data-Free-Quantization

2. Install pytorch and other dependencies.
    * pip install -r requirements.txt

### Set the paths of datasets for testing
1. Set the "dataPath" in "cifar100_resnet20.hocon" as the path root of your CIFAR-100 dataset. For example:

    dataPath = "/home/datasets/Datasets/cifar"

2. Set the "dataPath" in "imagenet_resnet18.hocon" as the path root of your ImageNet dataset. For example:

    dataPath = "/home/datasets/Datasets/imagenet"

### Training

To quantize the pretrained ResNet-20 on CIFAR-100 to 4-bit:

    python main.py --conf_path ./cifar100_resnet20.hocon --id 01
To quantize the pretrained ResNet-18 on ImageNet to 4-bit:

    python main.py --conf_path ./imagenet_resnet18.hocon --id 01

<br/>

## Results

|  Dataset | Model | Pretrain Top1 Acc(%) | W4A4(ours) Top1 Acc(%) |
   | :-: | :-: | :-: | :-: |
  | CIFAR-100 | ResNet-20| 70.33 | 63.58 ± 0.23 |
  | ImageNet | ResNet-18 | 71.47 | 60.60 ± 0.15 |

Note that we use the pretrained models from [pytorchcv](https://www.cnpython.com/pypi/pytorchcv).

<br/>

## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/2003.03603):

    @InProceedings{xu2020generative,
    title = {Generative Low-bitwidth Data Free Quantization},
    author = {Shoukai, Xu and Haokun, Li and Bohan, Zhuang and Jing, Liu and Jiezhang, Cao and Chuangrun, Liang and Mingkui, Tan},
    booktitle = {The European Conference on Computer Vision},
    year = {2020}
    }

<br/>

## Acknowledgments
This work was partially supported by Key-Area Research and Development Program of Guangdong Province (2019B010155002, 2018B010107001, 2019B010155-001), National Natural Science Foundation of China(NSFC) 61836003 (key project), 2017ZT07X183, Tencent AI Lab Rhino-Bird Focused Research Program (No.JR201902), Fundamental Research Funds for the Central Universities D2191240.