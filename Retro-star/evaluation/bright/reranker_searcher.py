import os
import json
import logging

from typing import Dict, Any, List, Tuple
from FlagEmbedding.abc.evaluation import EvalReranker

logger = logging.getLogger(__name__)


class ReasoningEvalReranker(EvalReranker):
    def __init__(
            self,
            reranker,
            reranker_scores_cache_dir: str,
            rerank_top_k: int = 100,
            multiple_pass_reranking_paths: List[str] = ["100@1"],
            reranker_enable_cache: bool = True,
            reranker_overwrite_cache: bool = False,
        ):
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

        self.reranker_scores_cache_dir = reranker_scores_cache_dir
        self.reranker_enable_cache = reranker_enable_cache
        self.reranker_overwrite_cache = reranker_overwrite_cache
        self.multiple_pass_reranking_paths = multiple_pass_reranking_paths

        self.reranker_sample_k = reranker.sample_k
        self.reranker_base_name = os.path.basename(self.reranker.model_name_or_path)

        current_pass = multiple_pass_reranking_paths[-1]
        top_k, sample_k = int(current_pass.split("@")[0]), int(current_pass.split("@")[1])
        if top_k != rerank_top_k or sample_k != self.reranker_sample_k:
            raise ValueError("The top_k and sample_k in the current pass must match the rerank_top_k and reranker.sample_k.")
        
        self.base_reranker_name = self.reranker_base_name

    def get_last_pass_running_name(self):
        pass_name = "-".join(self.multiple_pass_reranking_paths[:-1][::-1])
        if not pass_name:
            return None
        return pass_name + "-" + self.base_reranker_name

    def __str__(self) -> str:        
        pass_name = "-".join(self.multiple_pass_reranking_paths[::-1])
        return pass_name + "-" + self.base_reranker_name
    
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        search_results: Dict[str, Dict[str, float]],
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        running_time = 0.0

        excluded_ids = {}
        qrels = kwargs.get("reranker_qrels", None)
        if qrels is not None:
            for qid in qrels:
                excluded_ids[qid] = []
                for docid, score in qrels[qid].items():
                    if score != 1:
                        excluded_ids[qid].append(docid)
        else:
            logger.warning("No qrels provided, so no documents will be excluded.")
        
        # truncate search results to top_k
        for qid in search_results:
            # Filter out documents with ids in excluded_ids
            for doc_id in set(excluded_ids[qid]):
                if doc_id != "N/A":
                    search_results[qid].pop(doc_id, None)
            
            search_results[qid] = dict(
                sorted(search_results[qid].items(), key=lambda x: x[1], reverse=True)[
                    :self.rerank_top_k
                ]
            )
        # generate sentence pairs
        sentence_pairs = []
        pairs = []
        for qid in search_results:
            for docid in search_results[qid]:
                if ignore_identical_ids and qid == docid:
                    continue

                sentence_pairs.append(
                    {
                        "qid": qid,
                        "docid": docid,
                        "query": queries[qid],
                        "doc": corpus[docid]["text"] if "title" not in corpus[docid] 
                            else f"{corpus[docid]['title']} {corpus[docid]['text']}".strip(),
                    }
                )
                pairs.append(
                    (
                        queries[qid],
                        corpus[docid]["text"] if "title" not in corpus[docid] 
                            else f"{corpus[docid]['title']} {corpus[docid]['text']}".strip()
                    )
                )

        ##############################
        ### cache reranker scores
        ##############################
        if self.reranker_enable_cache:
            dataset_name = kwargs.get("dataset_name")
            cache_file = os.path.join(
                self.reranker_scores_cache_dir,
                self.base_reranker_name,
                f"{dataset_name}.json",
            )
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            if not self.reranker_overwrite_cache and os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    scores_cahce = json.load(f)
            else:
                scores_cahce = {}

            final_scores = [None] * len(sentence_pairs)
            new_pairs_to_compute, new_pairs_to_compute_indices = [], []

            sample_k_key = f"@{self.reranker_sample_k}"
            for i, pair_info in enumerate(sentence_pairs):
                q_d_key = f"{pair_info['qid']}-{pair_info['docid']}"
                if q_d_key in scores_cahce and sample_k_key in scores_cahce[q_d_key]:
                    final_scores[i] = scores_cahce[q_d_key][sample_k_key]
                else:
                    new_pairs_to_compute.append(pairs[i])
                    new_pairs_to_compute_indices.append(i)

            if_cache_update = False
            if new_pairs_to_compute:
                print(f"Computing scores for {len(new_pairs_to_compute)} pairs that are not in the cache.")
                new_scores, running_time, completion_tokens = self.reranker.compute_score(new_pairs_to_compute)
                for i, score in zip(new_pairs_to_compute_indices, new_scores):
                    final_scores[i] = float(score)
                    # update cache
                    q_d_key = f"{sentence_pairs[i]['qid']}-{sentence_pairs[i]['docid']}"
                    if q_d_key not in scores_cahce:
                        scores_cahce[q_d_key] = {}
                    scores_cahce[q_d_key][sample_k_key] = float(score)
                    if_cache_update = True

            if if_cache_update:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(scores_cahce, f, ensure_ascii=False, indent=4)
        ##############################
        ### cache reranker scores
        ##############################
        else:
            final_scores, running_time, completion_tokens = self.reranker.compute_score(pairs)
        
        for i, score in enumerate(final_scores):
            sentence_pairs[i]["score"] = float(score)
        
        reranked_results = {qid: {} for qid in search_results}
        for pair in sentence_pairs:
            reranked_results[pair["qid"]][pair["docid"]] = pair["score"]
        return reranked_results, running_time, completion_tokens
