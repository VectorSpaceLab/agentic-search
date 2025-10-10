<div align="center">
<h1> ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval </h1>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2510.08252" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/arXiv-2510.08252-B31B1B.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv:2510.08252">
  </a>
</p>


## Introduction

We propose **ReasonEmbed**, a new text embedding model for reasoning-intensive document retrieval based on innovations of how synthetic data is generated and used. Our work includes the following technical contributions.
1. We design a novel data synthesis method, called **ReMixer**.
2. We introduce a self-adaptive training method tailored for our synthetic data, termed **Redapter**.
3. We implement ReasonEmbed based on multiple LLM backbones of varying model sizes, which achieve **state-of-the-art (SOTA) performance** on reasoning-intensive document retrieval tasks. Notably, our model built on Qwen3-4B reaches an nDCG@10 score of 37.1 on the [BRIGHT](https://brightbenchmark.github.io/) benchmark, which already surpasses all existing text embedding models. While the Qwen3-8B based varient improves the performance to 38.1. Moreover, on the [R2MED](https://r2med.github.io/) benchmark, ReasonEmbed-Qwen3-8B attains an nDCG@10 score of 43.18, surpassing all of the existing models by a large margin and leading to new SOTA performance.

For more details, please refer to our [paper](https://arxiv.org/pdf/2510.08252).

## Open-Source Resources

### Model Checkpoints

TBA

### Evaluation Code and Scripts

For evaluation on the BRIGHT benchmark, refer to the [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/evaluation/bright) and [script](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/evaluation/bright/eval_bright_short.sh) provided by [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).

For evaluation on the R2MED benchmark, refer to the provided [code](https://github.com/VectorSpaceLab/agentic-search/tree/main/ReasonEmbed/evaluation_r2med/code) and [script](https://github.com/VectorSpaceLab/agentic-search/tree/main/ReasonEmbed/evaluation_r2med/scripts) (implemented using [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) framework).

### Data and Synthesis Code

TBA

### Training Code and Scripts

TBA


## Citation

If you find this repository useful, please consider giving a star ⭐ and citation:
```
@article{chen2025reasonembed,
  title={ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval},
  author={Chen, Jianlyu and Lan, Junwei and Li, Chaofan and Lian, Defu and Liu, Zheng},
  journal={arXiv preprint arXiv:2510.08252},
  year={2025}
}
```
