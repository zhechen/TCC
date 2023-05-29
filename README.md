# Transformer-based Context Condensation for Boosting Feature Pyramids in Object Detection (Accepted at IJCV)

By [Zhe Chen](https://scholar.google.cz/citations?user=Jgt6vEAAAAAJ&hl),  [Jing Zhang](https://scholar.google.com/citations?user=9jH5v74AAAAJ&hl), Yufei Xu, and [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl).

This repository is the implementation of the paper [TCC](https://arxiv.org/abs/2207.06603.pdf). 

This code is heavily based on a previous version (should be v2.28.2) of [MMDetection](https://github.com/open-mmlab/mmdetection.git).


## Introduction

![REGO](./figs/title.png)

**Abstract.** Current object detectors typically have a feature pyramid (FP) module for multi-level feature fusion (MFF) which aims to mitigate the gap between features from different levels and form a comprehensive object representation to achieve better detection performance. However, they usually require heavy cross-level connections or iterative refinement to obtain better MFF result, making them complicated in structure and inefficient in computation. To address these issues, we propose a novel and efficient context modeling mechanism that can help existing FPs deliver better MFF results while reducing the computational costs effectively. In particular, we introduce a novel insight that comprehensive contexts can be decomposed and condensed into two types of representations for higher efficiency. The two representations include a locally concentrated representation and a globally summarized representation, where the former focuses on extracting context cues from nearby areas while the latter extracts general contextual representations of the whole image scene as global context cues. By collecting the condensed contexts, we employ a Transformer decoder to investigate the relations between them and each local feature from the FP and then refine the MFF results accordingly. As a result, we obtain a simple and light-weight Transformer-based Context Condensation (TCC) module, which can boost various FPs and lower their computational costs simultaneously. Extensive experimental results on the challenging MS COCO dataset show that TCC is compatible to four representative FPs and consistently improves their detection accuracy by up to 7.8% in terms of average precision and reduce their complexities by up to around 20% in terms of GFLOPs, helping them achieve stateof-the-art performance more efficiently.


## Citing 
If you find our module useful in your research, please consider citing:
```bibtex
@article{chen2022transformer,
  title={Transformer-based Context Condensation for Boosting Feature Pyramids in Object Detection},
  author={Chen, Zhe and Zhang, Jing and Xu, Yufei and Tao, Dacheng},
  journal={arXiv preprint arXiv:2207.06603},
  year={2022}
}
```


## Installation

Please follow the [MMDetection](https://github.com/open-mmlab/mmdetection.git).

## Usage
Please see './configs/tcc' for the implementation of our method on the R50-FPN network.  


