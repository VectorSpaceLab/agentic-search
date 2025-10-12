# Reproduce Fusion Results

1. Download Search Results

```bash
git lfs install

git clone https://huggingface.co/datasets/hanhainebula/bright-search-results
```

2. Run Fusion

```bash
# Set the path to the downloaded search results
bright_search_results_root="./bright-search-results"

bash ./eval_bm25_reason-embed-basic-qwen3-8b-0928.sh $bright_search_results_root

bash ./eval_bge-reasoner-0928.sh $bright_search_results_root
```
