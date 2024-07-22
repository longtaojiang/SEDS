# SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval

#### Longtao Jiang<sup>1</sup>, Min Wang<sup>1</sup>, Zecheng Li<sup>1</sup>, Yao Fang<sup>2</sup>, Wengang Zhou<sup>1</sup>, Houqiang Li<sup>1</sup>.

<sup>1</sup> University of Science and Technology of China.
<sup>2</sup> Merchants Union Consumer Finance Company Limited.



## Abstract
Sign language retrieval, as an emerging visual-language task, has received widespread attention. Different from traditional video retrieval, it is more biased towards understanding the semantic information of human actions contained in video clips. Previous works typically only encode RGB videos to obtain high-level semantic features, resulting in local action details drowned in a large amount of visual information redundancy. Furthermore, existing RGB-based sign retrieval works suffer from the huge memory cost of dense visual data embedding in end-to-end training, and adopt offline RGB encoder instead, leading to suboptimal feature representation. To address these issues, we propose a novel sign language representation framework called Semantically Enhanced Dual-Stream Encoder (SEDS), which integrates Pose and RGB modalities to represent the local and global information of sign language videos. Specifically, the Pose encoder embeds the coordinates of keypoints corresponding to human joints, effectively capturing detailed action features. For better context-aware fusion of two video modalities, we propose a Cross Gloss Attention Fusion (CGAF) module to aggregate the adjacent clip features with similar semantic information from intra-modality and inter-modality. Moreover, a Pose-RGB Fine-grained Matching Objective is developed to enhance the aggregated fusion feature by contextual matching of fine-grained dual-stream features. Besides the offline RGB encoder, the whole framework only contains learnable lightweight networks, which can be trained end-to-end. Extensive experiments demonstrate that our framework significantly outperforms state-of-the-art methods on How2Sign, PHOENIX-2014T, and CSL-Daily datasets.


## Coming Soon


## Citation
```
Coming Soon
```

## Acknowledgment
The code is built based on [CiCo](https://github.com/FangyunWei/SLRT).
