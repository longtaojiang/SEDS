# SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval

<b> <a href='https://longtaojiang.github.io/'>Longtao Jiang</a>, Min Wang†, Zecheng Li, Yao Fang, <a href='http://staff.ustc.edu.cn/~zhwg/'>Wengang Zhou†</a>, <a href='http://staff.ustc.edu.cn/~lihq/en/'>Houqiang Li </a> </b>

[Paper](https://arxiv.org/abs/2407.16394) 

[Processed I3D Feature and RTM Keypoints](BaiduDrive (password: dire)(https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg) 

[pre-trained model](BaiduDrive (password: dire)(https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg)

## News
- [2024/09/26] :fire: Fix up the link of our data, model. We provide them in [[BaiduDrive (password: dire)](https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg)] and [[RecDrive (password: dire)](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070)].
- [2023/08/27] :fire: Release code, dataset and pre-trained models. [[OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EtKXrn4cjWtBi0H3v4j1ICsBKraCxnZiTWU4VzqRr0ilCw?e=trkgDR)]/[[RecDrive (code: dire)](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070)]

- [2023/07/14] :tada: DIRE is accepted by ICCV 2023.
- [2023/03/16] :sparkles: Release [paper](https://arxiv.org/abs/2303.09295).

## Abstract
> Sign language retrieval, as an emerging visual-language task, has received widespread attention. Different from traditional video retrieval, it is more biased towards understanding the semantic information of human actions contained in video clips. Previous works typically only encode RGB videos to obtain high-level semantic features, resulting in local action details drowned in a large amount of visual information redundancy. Furthermore, existing RGB-based sign retrieval works suffer from the huge memory cost of dense visual data embedding in end-to-end training, and adopt offline RGB encoder instead, leading to suboptimal feature representation. To address these issues, we propose a novel sign language representation framework called Semantically Enhanced Dual-Stream Encoder (SEDS), which integrates Pose and RGB modalities to represent the local and global information of sign language videos. Specifically, the Pose encoder embeds the coordinates of keypoints corresponding to human joints, effectively capturing detailed action features. For better context-aware fusion of two video modalities, we propose a Cross Gloss Attention Fusion (CGAF) module to aggregate the adjacent clip features with similar semantic information from intra-modality and inter-modality. Moreover, a Pose-RGB Fine-grained Matching Objective is developed to enhance the aggregated fusion feature by contextual matching of fine-grained dual-stream features. Besides the offline RGB encoder, the whole framework only contains learnable lightweight networks, which can be trained end-to-end. Extensive experiments demonstrate that our framework significantly outperforms state-of-the-art methods on How2Sign, PHOENIX-2014T, and CSL-Daily datasets.

## SEDS pipeline
<p align="center">
<img src="figs/dire.png" width=60%>
</p>

## Requirements
```
conda create -n dire python=3.9
conda activate dire
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Processed I3D Feature and RTM Keypoints
The DiffusionForensics dataset can be downloaded from [[BaiduDrive (password: dire)](https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg)] or [[RecDrive (password: dire)](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070)]. The dataset is organized as follows:
```
images/recons/dire
└── train/val/test
    ├── lsun_bedroom
    │   ├── real
    │   │   └──img1.png...
    │   ├── adm
    │   │   └──img1.png...
    │   ├── ...
    ├── imagenet
    │   ├── real
    │   │   └──img1.png...
    │   ├── adm
    │   │   └──img1.png...
    │   ├── ...
    └── celebahq
        ├── real
        │   └──img1.png...
        ├── adm
        │   └──img1.png...
        ├── ...

```
## Training
Before training, you should link the training real and DIRE images to the `data/train` folder. For example, you can link the DIRE images of real LSUN-Bedroom to `data/train/lsun_adm/0_real` and link the DIRE images of ADM-LSUN-Bedroom to `data/train/lsun_adm/1_fake`. And do the same for validation set and testing set, just modify `data/train` to `data/val` and `data/test`. Then, you can train the DIRE model by running the following command:
```
sh train.sh
```
## Evaluation
We provide the pre-trained DIRE model in [[BaiduDrive (password: dire)](https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg)] and [[RecDrive (password: dire)](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070)].
You can evaluate the DIRE model by running the following command:
```
sh test.sh
```
## Inference
We also provide a inference demo `demo.py`. You can run the following command to inference a single image or a folder of images:
```
python demo.py -f [image_path/image_dir] -m [model_path]
```

## Acknowledgments
Our code is developed based on [CiCo](https://github.com/FangyunWei/SLRT/tree/main/CiCo). Thanks for their sharing codes and models.

## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{10.1145/3664647.3681237,
author = {Jiang, Longtao and Wang, Min and Li, Zecheng and Fang, Yao and Zhou, Wengang and Li, Houqiang},
title = {SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681237},
doi = {10.1145/3664647.3681237},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {5141–5150},
numpages = {10},
keywords = {feature fusion, multimodal alignment, sign language retrieval},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

## Acknowledgment
The code is built based on [CiCo](https://github.com/FangyunWei/SLRT).
