# SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval

[Longtao Jiang](https://longtaojiang.github.io/), [Min Wang†], [Zecheng Li], [Yao Fang], [Wengang Zhou†](http://staff.ustc.edu.cn/~zhwg/), [Houqiang Li](http://staff.ustc.edu.cn/~lihq/en/),

(† Corresponding author)

Arxiv: [https://arxiv.org/abs/2407.16394](https://arxiv.org/abs/2407.16394)

## Abstract
> Sign language retrieval, as an emerging visual-language task, has received widespread attention. Different from traditional video retrieval, it is more biased towards understanding the semantic information of human actions contained in video clips. Previous works typically only encode RGB videos to obtain high-level semantic features, resulting in local action details drowned in a large amount of visual information redundancy. Furthermore, existing RGB-based sign retrieval works suffer from the huge memory cost of dense visual data embedding in end-to-end training, and adopt offline RGB encoder instead, leading to suboptimal feature representation. To address these issues, we propose a novel sign language representation framework called Semantically Enhanced Dual-Stream Encoder (SEDS), which integrates Pose and RGB modalities to represent the local and global information of sign language videos. Specifically, the Pose encoder embeds the coordinates of keypoints corresponding to human joints, effectively capturing detailed action features. For better context-aware fusion of two video modalities, we propose a Cross Gloss Attention Fusion (CGAF) module to aggregate the adjacent clip features with similar semantic information from intra-modality and inter-modality. Moreover, a Pose-RGB Fine-grained Matching Objective is developed to enhance the aggregated fusion feature by contextual matching of fine-grained dual-stream features. Besides the offline RGB encoder, the whole framework only contains learnable lightweight networks, which can be trained end-to-end. Extensive experiments demonstrate that our framework significantly outperforms state-of-the-art methods on How2Sign, PHOENIX-2014T, and CSL-Daily datasets.


## Coming Soon


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
