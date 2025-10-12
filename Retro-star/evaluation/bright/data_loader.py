import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


EXTENDED_SPLITS = [
    # w/ reasoning splits
]


class BrightShortEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return [
            # StackExchange
            "biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living",
            # Coding
            "leetcode", "pony",
            # Theorem-based
            "aops", "theoremqa_questions", "theoremqa_theorems"
        ]

    def available_splits(self, dataset_name: str) -> List[str]:
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ] + EXTENDED_SPLITS

    def _load_remote_corpus(
        self,
        dataset_name: str,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        corpus = datasets.load_dataset(
            "xlangai/bright", "documents",
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    docid, text = str(data["id"]), data["content"]
                    _data = {
                        "id": docid,
                        "text": text
                    }
                    corpus_dict[docid] = {"text": text}
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {str(data["id"]): {"text": data["content"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = 'examples',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        examples = datasets.load_dataset(
            "xlangai/bright", split,
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = defaultdict(dict)
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(examples, desc="Loading and Saving qrels"):
                    qid = f'{split}-{data["id"]}'
                    for docid in data["gold_ids"]:
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": 1
                        }
                        qrels_dict[qid][docid] = 1
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")

                    for ex_docid in list(set(data["excluded_ids"])):
                        if ex_docid == "N/A":
                            continue
                        assert ex_docid not in qrels_dict[qid], f"{ex_docid} in {qid}"
                        _data = {
                            "qid": qid,
                            "docid": ex_docid,
                            "relevance": 0
                        }
                        qrels_dict[qid][ex_docid] = 0
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
        else:
            qrels_dict = defaultdict(dict)
            for data in tqdm(examples, desc="Loading qrels"):
                qid = f'{split}-{data["id"]}'
                for docid in data["gold_ids"]:
                    qrels_dict[qid][docid] = 1
                
                for ex_docid in data["excluded_ids"]:
                    if ex_docid == "N/A":
                        continue
                    assert ex_docid not in qrels_dict[qid], f"{ex_docid} in {qid}"
                    _data = {
                        "qid": qid,
                        "docid": ex_docid,
                        "relevance": 0
                    }
                    qrels_dict[qid][ex_docid] = 0
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = 'examples',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        examples = datasets.load_dataset(
            "xlangai/bright", split,
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(examples, desc="Loading and Saving queries"):
                    qid, query = f'{split}-{data["id"]}', data["query"]
                    _data = {
                        "id": qid,
                        "text": query
                    }
                    queries_dict[qid] = query
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
        else:
            queries_dict = {f'{split}-{data["id"]}': data["query"] for data in tqdm(examples, desc="Loading queries")}
        return datasets.DatasetDict(queries_dict)
