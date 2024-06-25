# Generalizing-Grasp

**Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge**<br>

_Haoxiang Ma, Modishi, Boyang Gao, Di Huang_<br>
In CVPR'2024
#### [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf) [Video](https://www.youtube.com/watch?v=RzTXFcZURiU&t=14s)

## Introduction
This repository is official PyTorch implementation for our CVPR2024 paper.
The code is based on [GraspNet-baseline](https://github.com/graspnet/graspnet-baseline)

### Note: The repo is still updating

## Environments
- Anaconda3
- Python == 3.7.9
- PyTorch >= 1.8.0
- Open3D >= 0.8

## Installation
Follow the installation of graspnet-baseline.

Get the code.
```bash
git clone https://github.com/mahaoxiang822/Generalizing-Grasp.git
cd Generalizing-Grasp
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```


## Prepare Datasets
For GraspNet dataset, you can download from [GraspNet](https://graspnet.net)

#### Full scene data generation
You can generate fusion scene data by yourself by running:
```bash
cd scripts
python TSDFreconstruction_dataset.py
```
Or you can download the pre-generated data from [Google Drive](https://drive.google.com/file/d/12YODD0ZUu6XTudU1fZBhVtAmIpMZk8xQ/view?usp=sharing) and unzip it under the dataset root:

#### Object SDF generation
You can generate object SDF by running:
```bash
pip install mesh-to-sdf
python dataset/grid_sample.py
```

#### Tolerance Label Generation(Follow graspnet-baseline)
Tolerance labels are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_tolerance_label.py](../Scale-Balanced-Grasp/dataset/generate_tolerance_label.py). You can simply generate tolerance label by running the script: (`--dataset_root` and `--num_workers` should be specified according to your settings)
```bash
cd dataset
sh command_generate_tolerance_label.sh
```

Or you can download the tolerance labels from [Google Drive](https://drive.google.com/file/d/1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1HN29P-csHavJF-R_wec6SQ) and run:
```bash
mv tolerance.tar dataset/
cd dataset
tar -xvf tolerance.tar
```

## Train&Test

### Train with physical constrained regularization

```bash
sh command_train.sh
```

### Test
 - We offer our checkpoints for inference and evaluation, you can download from [Google Drive](https://drive.google.com/file/d/1WJj54l7MxFO1kgXoXA9tF6FCfB2okKr3/view?usp=sharing)
```bash
sh command_test.sh
```

- For contact-score joint optimization, first download the pretrained [contactnet](https://drive.google.com/file/d/1yMZ5rgloo0xbYvuR46t3sSKMvaVpOavx/view?usp=sharing) & [scorenet](https://drive.google.com/file/d/1didqsuweIbWb6UhL15IMhs2HrDhvC3EQ/view?usp=sharing) and unzip under the logs folder
then run:
```bash
sh optimization.sh
```
note: In current version, only the optimization with gt mask is uploaded and we will update the 3d segmentation version in the future.

#### Evaluation

```
python evaluate.py
```


### Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@InProceedings{Ma_2024_cvpr,
    author    = {Haoxiang, Ma and Modi, Shi and Boyang Gao and Di, Huang},
    title     = {Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
```