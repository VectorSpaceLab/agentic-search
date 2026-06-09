import os
import re
import json
import random
import argparse

import sglang as sgl

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from prompts import DatasetPrompts, PromptTemplate


class ReasoningEngine:
    def __init__(
        self,
        model_name_or_path: str,
        context_length: int = 32768,
        tp_size: int = 1,
        dp_size: int = 1,
        random_seed: int = 42,
        batch_size: int = 128,
    ):
        self.model_name_or_path = model_name_or_path
        self.context_length = context_length
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.random_seed = random_seed

        self.batch_size = batch_size
        self.sampling_params = self._get_sampling_params()

    def __del__(self):
        if hasattr(self, 'model'):
            self.model.shutdown()

    def _init_engine(self):
        self.model = sgl.Engine(
            model_path=self.model_name_or_path,
            context_length=self.context_length,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            random_seed=self.random_seed,
        )
        self.tokenizer = self.model.tokenizer_manager.tokenizer

    def _get_sampling_params(self):
        return {
            "n": 1,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "max_new_tokens": 2048,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            # "regex": r"(Yes|No)"
        }
    
    def reason(
        self,
        dataset: list,
        tmp_path: str,
    ) -> list:
        self._init_engine()
        query_level_responses = []

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # read temp (index, query_level_responses) from tmp_path
        # json format: {"index": 0, "query_level_responses": [{"query": "query1", "docs": ["doc1", "doc2"], "labels": [0, 1], "responses": ["Yes", "No"]}]}
        if os.path.exists(tmp_path):
            with open(tmp_path, 'r') as f:
                data = json.load(f)
            index = data["index"]
            query_level_responses = data["query_level_responses"]
            dataset = dataset[index:]
            print(f"[Load] Loaded {len(query_level_responses)} responses from {tmp_path}, continuing from index {index}.")

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Running pointwise reasoning", total=len(dataset) // self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batch_prompts, item_labels = [], []
            for item in batch:
                docs, labels = item["docs"], item["labels"]
                item_prompts = [
                    item["prompt_template"].format(
                        query=item["query"],
                        doc=doc,
                        query_type=item["query_type"],
                        doc_type=item["doc_type"],
                        relevance_definition=item["relevance_definition"]
                    )
                    for doc in docs
                ]
                batch_prompts.extend(item_prompts)
                item_labels.append(labels)

            messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
            input_texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            try:
                outputs = self.model.generate(
                    input_texts,
                    sampling_params=self.sampling_params,
                )
            except:
                outputs = [{"text": None} for _ in range(len(input_texts))]

            query_idx, query_output_idx = 0, 0
            while query_output_idx < len(outputs):
                query_outputs = outputs[query_output_idx:query_output_idx + len(item_labels[query_idx])]
                query_outputs = [output["text"] for output in query_outputs]

                query_level_responses.append({
                    "query": batch[query_idx]["query"],
                    "docs": batch[query_idx]["docs"],
                    "labels": item_labels[query_idx],
                    "responses": query_outputs,
                })

                query_output_idx += len(item_labels[query_idx])
                query_idx += 1

            # Save the current index and responses to tmp_path
            with open(tmp_path, 'w') as f:
                json.dump({
                    "index": i + self.batch_size,
                    "query_level_responses": query_level_responses
                }, f, ensure_ascii=False, indent=4)
            print(f"[Save] Saved {len(query_level_responses)} responses to {tmp_path} at index {i + self.batch_size}.")
        return query_level_responses
    

class TrainingDatasetBuilder:
    def __init__(
        self,
        data_file: str,
        cache_dir: str,
        output_name: str,
        task_type: str,
        output_dir: str,
        top_k: int = 100,
        random_seed: int = 42,
    ):
        self.data_file = data_file
        self.cache_dir = cache_dir
        self.output_name = output_name
        self.task_type = task_type
        self.output_dir = output_dir
        self.top_k = top_k
        self.random_seed = random_seed

    def _load_raw_dataset(self):
        raw_dataset = load_dataset(
            "json",
            data_files=self.data_file,
            cache_dir=self.cache_dir,
        )["train"]
        raw_dataset = raw_dataset.shuffle(seed=self.random_seed)
        return raw_dataset
    
    def _get_prompt_template(self):
        return PromptTemplate
    
    def _get_dataset_prompts(self):
        return DatasetPrompts[self.task_type]
    
    def _transform_raw_dataset(self, raw_dataset: Dataset, drop_length_threshold: int = -1) -> list:
        dataset = []
        prompt_template = self._get_prompt_template()
        query_type, doc_type, relevance_definition = self._get_dataset_prompts()
        for item in tqdm(raw_dataset, desc="Transforming raw dataset"):
            query = item["query"]
            
            if drop_length_threshold > 0 and len(query) > drop_length_threshold * 0.3:
                print(f"[Drop] Query '{query}' exceeds length threshold, skipping.")
                continue
            
            docs = []
            for doc in item["docs"]:
                if drop_length_threshold > 0 and len(doc) > drop_length_threshold * 0.3:
                    print(f"[Drop] Document exceeds length threshold, skipping.")
                    continue
                docs.append(doc)
                if len(docs) >= self.top_k:
                    break
            if len(docs) == 0:
                print(f"[Drop] No valid documents found for query '{query}', skipping.")
                continue
            
            labels = [0] * len(docs)  # Initialize labels with 0, will be updated later
            data = {
                "query": query,
                "prompt_template": prompt_template,
                "query_type": query_type,
                "doc_type": doc_type,
                "relevance_definition": relevance_definition,
                "docs": docs,
                "labels": labels,
            }
            dataset.append(data)
        return dataset
    
    def _get_temp_path(self):
        return os.path.join(
            self.output_dir,
            "tmp-label-query",
            self.output_name,
            f"{os.path.basename(self.data_file).replace('-to-be-labeled.jsonl', '')}-{os.path.basename(self.engine.model_name_or_path)}.json",
        )

    def _save_label_dataset(self, pointwise_sft_dataset):
        output_sft_file = os.path.join(
            self.output_dir,
            self.output_name,
            f"{os.path.basename(self.data_file).replace('-to-be-labeled.jsonl', '')}-{os.path.basename(self.engine.model_name_or_path)}-labeled.jsonl",
        )
        os.makedirs(os.path.dirname(output_sft_file), exist_ok=True)

        updated_sft_dataset = []
        for item in pointwise_sft_dataset:
            query, docs, labels = item["query"], item["docs"], item["labels"]
            updated_sft_dataset.append({
                "query": query,
                "docs": docs,
                "labels": labels,
            })

        Dataset.from_list(updated_sft_dataset).to_json(
            output_sft_file,
            lines=True,
            force_ascii=False,
        )
        print(f"[Save] Saved {len(updated_sft_dataset)} samples to {output_sft_file}")

    def _label(self, dataset, query_level_responses):
        pointwise_label_dataset = []
        for item, query_level_response in tqdm(zip(dataset, query_level_responses), desc="Reject Sampling Pointwise"):
            assert item["query"] == query_level_response["query"], "Query mismatch between dataset and model response."
            labels, responses = item["labels"], query_level_response["responses"]
            
            for idx, response in enumerate(responses):
                try:
                    score = int(re.search(r"<score>\s*(\d+)\s*</score>", response).group(1))
                except:
                    score = -1  # Invalid score
                
                labels[idx] = score
            
            pointwise_label_dataset.append(item | {"labels": labels})

        print(f"[Filter] Filtered {len(pointwise_label_dataset)}({len(pointwise_label_dataset) / len(dataset):.2%}) pointwise samples with positive labels.")
        return pointwise_label_dataset

    def run(self, engine: ReasoningEngine, drop_length_threshold: int = -1):
        self.engine = engine
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.engine.model_name_or_path,
        )
        output_sft_file = os.path.join(
            self.output_dir,
            self.output_name,
            f"{os.path.basename(self.data_file).replace('-to-be-labeled.jsonl', '')}-{os.path.basename(self.engine.model_name_or_path)}-labeled.jsonl",
        )
        if os.path.exists(output_sft_file):
            print(f"[Skip] {output_sft_file} already exists, skipping labeling.")
            return

        raw_dataset = self._load_raw_dataset()
        dataset = self._transform_raw_dataset(raw_dataset, drop_length_threshold=drop_length_threshold)

        dataset_model_response = self.engine.reason(dataset, self._get_temp_path())
        pointwise_label_dataset = self._label(dataset, dataset_model_response)
        self._save_label_dataset(pointwise_label_dataset)
        

def get_args():
    parser = argparse.ArgumentParser(description="Reject Sampling with LLM Scorer")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the SGLang model")
    parser.add_argument("--context_length", type=int, default=32768, help="Context length for the model")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing the dataset")

    parser.add_argument("--data_file", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_DATASETS_CACHE") or os.environ.get("CACHE_DIR"), help="Directory to cache datasets")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output dataset")
    parser.add_argument("--task_type", type=str, required=True, help="")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for output files")

    parser.add_argument("--top_k", type=int, default=100, help="Top-k sampling parameter")

    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    random.seed(args.seed)

    engine = ReasoningEngine(
        model_name_or_path=args.model_name_or_path,
        context_length=args.context_length,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        random_seed=args.seed,
        batch_size=args.batch_size,
    )

    builder = TrainingDatasetBuilder(
        data_file=args.data_file,
        cache_dir=args.cache_dir,
        output_name=args.output_name,
        task_type=args.task_type,
        output_dir=args.output_dir,
        top_k=args.top_k,
        random_seed=args.seed,
    )
    
    builder.run(engine, drop_length_threshold=args.context_length * 4)


if __name__ == "__main__":
    args = get_args()
    main(args)
