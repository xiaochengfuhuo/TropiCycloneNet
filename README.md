# TropiCycloneNet
![图片](https://github.com/user-attachments/assets/792867ad-5ea4-49c8-beb9-267276de7aec)
This project includes a subset of our global TC track and intensity prediction dataset $TCN_{D}$ and the test code of our deep learning TC forecasting method $TCN_{M}$.

**The training code is released now**

## Introduction

To show our work intuitively, we provide this code to visualize our track prediction results on the Himawari-8 satellite cloud image. (Our method can also provide the TC intensity predictions. But there is not a very suitable way to show them so this code only shows the track prediction results)
![Sample](LINGLING.gif)
In the above picture, the sequence of red circles depicts the actual trajectory, while the semi-transparent green area illustrates the potential trends derived from our multiple trajectory predictions. Similarly, the semi-transparent red area indicates the potential trends according to MMSTN. The sequence of green stars represents the most accurate prediction trajectory produced by our method. Additionally, the backdrop for these prediction results features the satellite cloud imagery for each tropical cyclone (TC).

## Requirements 
* python 3.8.5
* Pytorch 1.11.0 (GPU)

## Data Preparation
First, we need to download all the data we used in TropiCycloneNet.
* $TCN_{D}$'s [subset](https://drive.google.com/file/d/1YJg_gjF-zqvRdNpmAWFG4bG0Akwv_r2p/view?usp=sharing)
* Himawari-8 satellite [cloud image](https://drive.google.com/file/d/1xg6xxYxO_ASkI54C8tPyhNS1ceyrfv5b/view?usp=sharing)
* $TCN_{M}$'s [checkpoint](https://drive.google.com/file/d/1j5r2L5Y5W81pn7nBfrZCT1BA1_qfnaay/view?usp=sharing)

After completing the downloading, there are some files.
As for $TCN_{D}$'s subset, it includes BST data ($Data_{1d}$), a part of ERA5 data($Data_{2d}$, GPH 500 hPa), and Environment data($Env-data$). You can extract them somewhere you like.

As for the Himawari-8 satellite cloud image, it will be used as the background of our track prediction results.

As for $TCN_{M}$'s checkpoint, it is our well-trained model, you need to move it to **\scripts\model_save\best** before you run our code.

## Test
```python
## Visualize some samples##
cd scripts
python visual_evaluate_model_Me.py --TC_name MALIKSI --TC_date 2018061006  --TC_img_path [Himawari-8 satellite cloud image path] --TC_data_path [$TCN_{D}$'s subset path]
```
**TC_name** and **TC_date** are the parameters presenting the TC you want to predict. You can change it and see some other predictions. Please check the TC in the folder **Himawari_airmass** and choose the TC to predict (at this moment, we just provided cloud images in the year 2018 and 2019).

After running the code (about 1 min), you can check the results at **\scripts\plot**

## Training

First, download the [TCND dataset](https://zenodo.org/records/17104690), which has already been preprocessed and is ready for training. Then, extract the `Data_1d`, `Data_3d`, and `Env-Data` folders into their respective directories.

Next, update the following file paths in the code:

* In `scripts/train_github.py`, modify line 36 to point to the path of `Data_1d`.
* In `TCNM/data/trajectoriesWithMe_unet_training.py`, modify line 379 to point to the path of `Data_3d`, and line 334 to point to the path of `Env-Data`.

Once that's done, run the command below to start training:

```bash
## model training ##
cd scripts
python train_github.py
```


## Citing TropiCycloneNet

```
@article{Huang2025,
  author    = {Huang, Cheng and Mu, Pan and Zhang, Jinglin and Chan, Sixian and Zhang, Shiqi and Yan, Hanting and Chen, Shengyong and Bai, Cong},
  title     = {Benchmark dataset and deep learning method for global tropical cyclone forecasting},
  journal   = {Nature Communications},
  volume    = {16},
  number    = {1},
  pages     = {5923},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41467-025-61087-4},
  url       = {https://doi.org/10.1038/s41467-025-61087-4},
  issn      = {2041-1723}
}
```
