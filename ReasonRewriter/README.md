# BGE-Rewriter

## Introduction

We propose **ReasonRewriter**, a rewriting model designed for reasoning-intensive document retrieval tasks. It rewrites queries into more suitable forms to enhance retrieval performance.

## Open-source Resources

### Model

| Resource Type | Name                       | Link                                                         | Release Date | Comments                                                     |
| ------------- | -------------------------- | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| Model         | BGE-Reasoner-Rewriter-0821 | 🤗[reasoner-rewriter-qwen2.5-7b-0821](https://huggingface.co/cfli/reasoner-rewriter-qwen2.5-7b-0821) | Oct 11, 2025 | fine-tuned on [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with BGE-Reasoner-Data-0904 |

### Data

| Resource Type   | Name                              | Link                                                         | Release Date | Comments |
| --------------- | --------------------------------- | ------------------------------------------------------------ | ------------ | -------- |
| Rewritten Query | BGE-Reasoner-Rewritten-Query-0821 | 🤗[reasoner-rewritten-query-0821](https://huggingface.co/datasets/cfli/reasoner-rewritten-query-0821) | Oct 11, 2025 |          |

## Evaluation

### For BM25

Prepare Environment

```bash
conda create -n bright python=3.10
conda activate bright
git clone https://github.com/xlang-ai/BRIGHT
cd BRIGHT
conda install -n bright -c conda-forge openjdk=22
sudo dpkg -i
pip install -r requirements.txt
cd ..
```

Eval

```bash
rm ./BRIGHT/retrievers.py
rm ./BRIGHT/run.py
cp ./evaluation/BRIGHT-offical/retrievers.py ./BRIGHT
cp ./evaluation/BRIGHT-offical/run.py ./BRIGHT
cd ./BRIGHT
python run.py --task {task} --model bm25
```

### For Other models

Prepare Environment

```bash
conda create -n flagembedding python=3.10
conda activate flagembedding
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -r requirements.txt
pip install pytrec_eval
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd ..
```

Eval

```bash
rm ./FlagEmbedding/FlagEmbedding/evaluation/bright/search.py
rm ./FlagEmbedding/FlagEmbedding/evaluation/bright/data_loader.py
cp ./FlagEmbedding/FlagEmbedding/evaluation/bright/search.py ./FlagEmbedding/FlagEmbedding/evaluation/bright
cp ./FlagEmbedding/FlagEmbedding/evaluation/bright/data_loader.py ./FlagEmbedding/FlagEmbedding/evaluation/bright
```

then using scripts from [embedder-scripts](https://github.com/VectorSpaceLab/agentic-search/tree/main/ReasonEmbed/evaluation_bright/scripts)