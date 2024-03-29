# TropiCycloneNet
This project includes a subset of our global TC track and intensity prediction [dataset](https://drive.google.com/file/d/1AZhiGk-cTvcMdL9jerL25KmTzqD8Kab-/view?usp=sharing) $TCN_{D}$ and the test code of our deep learning TC forecasting method $TCN_{M}$.

The training code will come soon.

## Introduction

To show our work intuitively, we provide this code to visualize our track prediction results on the Himawari-8 satellite cloud image. (Our method can also provide the TC intensity predictions. But there is not very suitable way to show them so this code only shows the track prediction results)
[Sample]


## Requirements 
* python 3.8.5
* Pytorch 1.11.0
* CUDA 11.7

## Data Preparation
First, you need to download all the data that we used in MGTCF by [Baidu Netdisk](https://pan.baidu.com/s/1qLEVymQ3ivvqAbgGBNkgaQ?pwd=rgwn ) and [Google Drive](https://drive.google.com/file/d/1AZhiGk-cTvcMdL9jerL25KmTzqD8Kab-/view?usp=sharing).

After you complete the downloading, you will find there is a compressed file, which includes Data_1d (Folder name: **1950_2019**), Data_2d (Folder name: **geopotential_500_year_centercrop**), Env-Data (Folder name: **env_data**), and a documentary (file name: [**README.pdf**](https://github.com/Zjut-MultimediaPlus/MGTCF/blob/main/README.pdf)) including some details about these data.

Then, you need to move the Folder **1950_2019** to the **datasets** under the main folder of this project and the **datasets** under the folder **scripts** and correct the path of **geopotential_500_year_centercrop** and the **env_data** in line 302(env_data) and line 309(Data_2d). 

## Train
```python
##before train##
python -m visdom.server
##custom train##
python train.py
```
## Test
```python
## test on data of the year 2019##
python evaluate_model_ME.py --dset_type test2019
```
## Training new models
Instructions for training new models can be [found here](https://github.com/Zjut-MultimediaPlus/MGTCF/blob/main/TRAINING.md).

## The data we used
We used two open-access dataset: [the CMA Tropical Cyclone Best Track Dataset](https://tcdata.typhoon.org.cn/en/zjljsjj_sm.html) 
, [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) 
and the results of [the CMO's tropical cyclone predictions](http://typhoon.nmc.cn/web.html).

If you want to use these data, we upload them on [Baidu Netdisk](https://pan.baidu.com/s/1qLEVymQ3ivvqAbgGBNkgaQ?pwd=rgwn ) and [Google Drive](https://drive.google.com/file/d/1AZhiGk-cTvcMdL9jerL25KmTzqD8Kab-/view?usp=sharing).
.

If you are interested in these data, you can click [the CMA Tropical Cyclone Best Track Dataset](https://tcdata.typhoon.org.cn/en/zjljsjj_sm.html), [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5), and
[the CMO's tropical cyclone data](http://typhoon.nmc.cn/web.html) to obtain more details. 



## Note
Although MGTCF gets a surprising performance, there are some challenges in the task of tropical cyclone forecasting. First, the performance of trajectory prediction in long term is still worse than the performance of the official meteorological agencies. Second, due to the use of ERA5, which is reanalysis data, MGTCF can not predict TC in real-time. It is a critical problem that determines whether our research can be implemented in industry. Therefore, the next step is to continue to improve the performance of forecasting and find a solution to make our method predict TC in real-time.

Our codes were modified from the implementation of ["MMSTN: a Multi-Modal Spatial-Temporal Network for Tropical Cyclone Short-Term Prediction"](https://github.com/Zjut-MultimediaPlus/MMSTN). Please cite the two papers (MGTCF and MMSTN) when you use the codes.
## Citing MGTCF & MMSTN
```
@inproceedings{MGTCF,
  title={MGTCF: Multi-Generator Tropical Cyclone Forecasting with Heterogeneous Meteorological Data},
  author={Huang, Cheng and Bai, Cong and Chan, Sixian and Zhang, Jinglin and Wu, YuQuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={4},
  pages={5096--5104},
  year={2023}
}
```

```
@article{https://doi.org/10.1029/2021GL096898,
author = {Huang, Cheng and Bai, Cong and Chan, Sixian and Zhang, Jinglin},
title = {MMSTN: A Multi-Modal Spatial-Temporal Network for Tropical Cyclone Short-Term Prediction},
journal = {Geophysical Research Letters},
volume = {49},
number = {4},
pages = {e2021GL096898},
doi = {https://doi.org/10.1029/2021GL096898},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021GL096898},
year = {2022}
}
```
