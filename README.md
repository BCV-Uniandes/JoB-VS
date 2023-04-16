# JoB-VS

This repository provides a PyTorch implementation of [JoB-VS: Joint Brain-Vessel Segmentation in TOF-MRA Images](https://arxiv.org/abs/2107.04263) presented in [ISBI 2023](https://2023.biomedicalimaging.org/en/default.asp). JoB-VS performs a joint-task learning for brain and vessel segmentation (JoB-VS) in Time-of-Flight Magnetic Resonance images, being an end-to-end vessel segmentation framework. Unlike other vessel segmentation methods, our approach avoids the pre-processing step of implementing a model to extract the brain from the volumetric input data. Our method builds upon [Towards Robust General Medical Image Segmentation](https://arxiv.org/abs/2107.04263) with a segmentation head that allows the simultaneous prediction of the brain and vessel mask.

## Paper

[JoB-VS: Joint Brain-Vessel Segmentation in TOF-MRA Images]https://biomedicalcomputervision.uniandes.edu.co/wp-content/uploads/2023/04/ISBI2023_paper_07517553.pdf),<br/>
Natalia Valderrama<sup>1</sup>, Ioannis Pitsiorlas<sup>2</sup>, Luisa Vargas<sup>1</sup>,[Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup>*<br/>, Maria A. Zuluaga<sup>2</sup>
[ISBI 2023](https://2023.biomedicalimaging.org/en/default.asp).<br><br>
<sup>1 </sup> Center for Research and Formation in Artificial Intelligence ([CINFONIA](https://cinfonia.uniandes.edu.co/)), Universidad de Los Andes. <br/>
<sup>2 </sup>Data Science Department, EURECOM, Sophia Antipolis, France<br/>

## Installation

### Cloning the repository

```bash
$ git clone git@github.com:BCV-Uniandes/ROG.git
$ cd ROG
$ python setup.py install
```
## Dataset Preparation

1. Download your data and create a json file in the OASIS3 format. [Here](https://github.com/BCV-Uniandes/JoB-VS/tree/main/datasets) you can find an example of how the data must be organized. Specify the root to your data in the json file.

2. Set the `data_root`, `out_directory` and `num_workers` variables in the file [`data_preprocessing.py`](https://github.com/BCV-Uniandes/JoB-VS/tree/main/libs/preprocessing) and run the command:

```
python data_preprocessing.py
```

Your data will be organized in the following way:
```

Fold_X
|_ imagesTr
|_ |_ *.nii.gz
|_ imagesTs
|_ |_ *.nii.gz
|_ labelsTr
|_ |_ *.nii.gz
|_ dataset.json
|_ dataset_stats.json
```
Our benchmark is setup for 2 folds. 

2. (optional) If data doesn't have any labels, as in the IXI dataset, please use this file [`data_preprocessing.py`](https://github.com/BCV-Uniandes/JoB-VS/tree/main/libs/preprocessing_ixi).


## Training and evaluating the models

We train JoB-VS on the original images, without using brain masks, and then we fine-tune the models using Free AT, as done in ROG:

```
# For the training on original images
python main.py --gpu GPU_IDs --batch BATCH_SIZE --data_ver OUT_DIRECTORY --name OUTPUT_DIR

# For the Free AT fine tuning
python main.py --gpu GPU_IDs --batch BATCH_SIZE --data_ver OUT_DIRECTORY --name OUTPUT_DIR_FREE_AT --ft --pretrained OUTPUT_DIR --AT
```

For evaluating the models, modify the EXPS_PATH, PATH_ANNS and PATH_PREDS in the file run_evaluations.py:

```
# Standard inference
python run_evaluations.py

```

If you want to make inference with our models, please download our weights in this [link](https://drive.google.com/drive/folders/1CtjMdQ5Ip1zCjKZlPDVGvp_r0rBv218w?usp=sharing) and run:

```
# For the training on original images
python main.py --gpu GPU_IDs --batch BATCH_SIZE --data_ver YOUR_DATA --name OUTPUT_DIR --load_weights WEIGHTS_PATH --test

```

If you are using the ixi dataset, please add the (`--ixi`) flag.

## MONAILabel APP

Please find all the information for the MONAILabel app in the branch monai.