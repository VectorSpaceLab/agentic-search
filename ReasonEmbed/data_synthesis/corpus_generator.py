import os
import random
import datasets
from tqdm import tqdm

from constant import DOC_LENGTH_THRESHOLD


BRIGHT_DATA_ROOT = os.environ.get("BRIGHT_DATA_ROOT", "./bright_short/data")


class CorpusGenerator:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
    
    def _load_corpus(self, corpus_path: str, qrels_path: str):
        assert corpus_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."
        assert qrels_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."
        
        # load corpus
        corpus = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']
        
        # load qrels
        qrels = datasets.load_dataset('json', data_files=qrels_path, cache_dir=self.cache_dir)['train']
        skip_docids = set()
        for qrel in qrels:
            skip_docids.add(qrel["docid"])
        
        # docs need to be skipped
        skip_docs = set()
        
        corpus_list = []
        for data in tqdm(corpus, desc="Loading corpus"):
            if data["id"] not in skip_docids:
                text = data["text"]
                
                # check document length
                min_threshold, max_threshold = DOC_LENGTH_THRESHOLD
                if len(text) < min_threshold or len(text) > max_threshold:
                    continue
                
                if text not in skip_docs and text.strip() not in skip_docs:
                    corpus_list.append({
                        "text": text,
                    })

        return corpus_list
    
    def run(
        self,
        task_type: str,
        num_samples: int = -1,
        use_cleaned_corpus: bool = False,
    ):
        if use_cleaned_corpus:
            corpus_path = os.path.join(BRIGHT_DATA_ROOT, task_type, "annotated_for_generation", "corpus_annotated_true.jsonl")
        else:
            corpus_path = os.path.join(BRIGHT_DATA_ROOT, task_type, "corpus.jsonl")
        qrels_path = os.path.join(BRIGHT_DATA_ROOT, task_type, "examples_qrels.jsonl")
        
        corpus_list = self._load_corpus(corpus_path, qrels_path)
        
        if num_samples > 0 and num_samples < len(corpus_list):
            corpus_list = random.sample(corpus_list, num_samples)
        
        return corpus_list
