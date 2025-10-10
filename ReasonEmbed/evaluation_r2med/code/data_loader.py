import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class R2MEDEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return [
            "Biology",
            "Bioinformatics",
            "Medical-Sciences",
            "MedXpertQA-Exam",
            "MedQA-Diag",
            "PMC-Treatment",
            "PMC-Clinical",
            "IIYi-Clinical",
        ]

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        return ["test"]

    def _load_remote_corpus(
        self,
        dataset_name: str,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the corpus dataset from HF.

        Args:
            dataset_name (str): Name of the dataset.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of corpus.
        """
        corpus = datasets.load_dataset(
            f"R2MED/{dataset_name}", "corpus",
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )["corpus"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    docid, text = str(data["id"]), data["text"]
                    _data = {
                        "id": docid,
                        "text": text
                    }
                    corpus_dict[docid] = {
                        "text": text
                    }
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {str(data["docid"]): {"text": data["text"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        qrels = datasets.load_dataset(
            f"R2MED/{dataset_name}", "qrels",
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )["qrels"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(qrels, desc="Loading and Saving qrels"):
                    qid, docid, rel = str(data["q_id"]), str(data["p_id"]), int(data["score"])
                    _data = {
                        "qid": qid,
                        "docid": docid,
                        "relevance": rel
                    }
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid][docid] = rel
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} qrels saved to {save_path}")
        else:
            qrels_dict = {}
            for data in tqdm(qrels, desc="Loading qrels"):
                qid, docid, rel = str(data["q_id"]), str(data["p_id"]), int(data["score"])
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][docid] = rel
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        queries = datasets.load_dataset(
            f"R2MED/{dataset_name}", "query",
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )["query"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(queries, desc="Loading and Saving queries"):
                    qid, query = str(data["id"]), data["text"]
                    qid = str(qid)
                    _data = {
                        "id": qid,
                        "text": query
                    }
                    queries_dict[qid] = query
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} queries saved to {save_path}")
        else:
            queries_dict = {str(data["id"]): {"text": data["text"]} for data in tqdm(queries, desc="Loading queries")}
        return datasets.DatasetDict(queries_dict)

