import os
import json
import argparse
from collections import defaultdict

from FlagEmbedding.abc.evaluation import AbsEvaluator
from FlagEmbedding.evaluation.bright import BrightShortEvalDataLoader


def load_jsonl_data(file_path: str):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            data_list.append(data)
    return data_list


def load_json_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data_list_to_jsonl(data_list: list, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"Saved {len(data_list)} samples to {file_path}")


def save_data_to_json(data, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {file_path}")


def load_bright_data_loader():
    data_loader = BrightShortEvalDataLoader(
        eval_name="bright_short",
        dataset_dir="/share/project/jianlv/o1-Embedder-Extension/evaluation_embedder/short/data",
        cache_dir="/share/project/jianlv/datasets/.cache",
    )
    return data_loader


def load_bright_evaluator(data_loder: BrightShortEvalDataLoader):
    evaluator = AbsEvaluator(
        eval_name="bright_short",
        data_loader=data_loder,
    )
    return evaluator


def init_weights(weights, num_results_dirs, norm_weights=False):
    if weights is None:
        weights = [1.0 / num_results_dirs] * num_results_dirs
    elif isinstance(weights, list):
        if len(weights) == 1:
            weights = weights * num_results_dirs
        assert len(weights) == num_results_dirs, "Weights must match the number of results directories."
        if norm_weights:
            weights = [w / sum(weights) for w in weights]
    elif isinstance(weights, float):
        weights = [weights] * num_results_dirs
        weights = [w / sum(weights) for w in weights]
    else:
        raise ValueError("Weights must be a list or a float.")
    return weights


def min_max_norm(score_dict: dict):
    min_score = min(score_dict.values())
    max_score = max(score_dict.values())
    
    if min_score == max_score:
        return {k: 1.0 for k in score_dict.keys()}
    
    normed_scores = {k: (v - min_score) / (max_score - min_score) for k, v in score_dict.items()}
    return normed_scores


def load_search_results(results_dir: str, domain: str, split_name: str, norm: bool = True, top_k: int = 100):
    search_results_path = os.path.join(results_dir, f"{domain}-{split_name}.json")
    if not os.path.exists(search_results_path):
        raise FileNotFoundError(f"Search results file not found: {search_results_path}")
    search_results = load_json_data(search_results_path)["search_results"]
    
    # truncate to top_k
    assert top_k > 0, "top_k must be greater than 0."
    search_results = {qid: dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]) for qid, doc_scores in search_results.items()}
    
    if not norm:
        return search_results
    else:
        new_search_results = {}
        for qid, doc_scores in search_results.items():
            new_doc_scores = min_max_norm(doc_scores) if norm else doc_scores
            new_search_results[qid] = new_doc_scores
        return new_search_results


def get_rrf_score(doc_scores: dict, rrf_k: int = 60):
    rrf_scores = {}
    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
        rrf_scores[doc_id] = 1 / (rrf_k + rank)
    return rrf_scores


def merge_search_results(search_results_list, weights, ori_split_name, use_rrf, rrf_k):
    merged_search_results = defaultdict(lambda: defaultdict(float))
    for search_results, weight in zip(search_results_list, weights):
        for qid, doc_scores in search_results.items():
            # replace the original split name with "examples"
            qid = qid.replace(ori_split_name, "examples")
            
            if use_rrf: 
                doc_scores = get_rrf_score(doc_scores, rrf_k=rrf_k)
            
            for doc_id, score in doc_scores.items():
                merged_search_results[qid][doc_id] += score * weight
    return merged_search_results


def compute_average_results(eval_results: dict):
    avg_eval_results_list = defaultdict(list)
    for _, results_dict in eval_results.items():
        for k, v in results_dict.items():
            avg_eval_results_list[k].append(v)
    
    avg_eval_results = {}
    for k, v_list in avg_eval_results_list.items():
        avg_eval_results[k] = sum(v_list) / len(v_list) if v_list else 0.0
    return avg_eval_results


def init_merged_model_name(merged_model_name, weights, norm, top_k, use_rrf, rrf_k):
    if not use_rrf:
        new_merged_model_name = f"{merged_model_name}-top_k_{top_k}-norm_{norm}-weights_" + "_".join([f"{w:.2f}" for w in weights])
    else:
        new_merged_model_name = f"{merged_model_name}-top_k_{top_k}-norm_{norm}-rrf_k_{rrf_k}-weights_" + "_".join([f"{w:.2f}" for w in weights])
    return new_merged_model_name


def main(args):
    data_loader = load_bright_data_loader()
    evaluator = load_bright_evaluator(data_loader)
    
    norm = args.norm
    results_dirs = args.results_dirs
    weights = args.weights
    norm_weights = args.norm_weights
    merged_model_name = args.merged_model_name
    ori_split_name = args.ori_split_name
    top_k = args.top_k
    
    use_rrf = args.use_rrf
    rrf_k = args.rrf_k
    
    only_transfer = args.only_transfer
    
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    
    weights = init_weights(weights, len(results_dirs), norm_weights=norm_weights)
    
    if not only_transfer:
        merged_model_name = init_merged_model_name(merged_model_name, weights, norm, top_k, use_rrf, rrf_k)
    
    save_dir = os.path.join(args.save_dir, merged_model_name, "NoReranker")
    os.makedirs(save_dir, exist_ok=True)
    
    domain_list = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "leetcode", "pony", "aops", "theoremqa_questions", "theoremqa_theorems"]
    for domain in domain_list:
        search_results_list = [
            load_search_results(results_dir, domain, ori_split_name, norm, top_k) for results_dir in results_dirs
        ]
        merged_search_results = merge_search_results(search_results_list, weights, ori_split_name, use_rrf, rrf_k)
        
        results_dict = {
            "eval_name": "bright_short",
            "model_name": merged_model_name,
            "reranker_name": "NoReranker",
            "split": "examples",
            "dataset_name": domain,
            "search_results": merged_search_results,
        }
        save_path = os.path.join(save_dir, f"{domain}-examples.json")
        
        save_data_to_json(results_dict, save_path)
    
    eval_save_dir = os.path.join(save_dir, "EVAL")
    os.makedirs(eval_save_dir, exist_ok=True)
    
    eval_save_path = os.path.join(eval_save_dir, "eval_results.json")
    eval_results = evaluator.evaluate_results(save_dir, k_values=[1, 10, 100])
    save_data_to_json(eval_results, eval_save_path)
    domain_results = [eval_results[f"{domain}-examples"]["ndcg_at_10"] * 100 for domain in domain_list]
    
    # compute the average reesults
    avg_eval_save_path = os.path.join(eval_save_dir, "avg_eval_results.json")
    avg_eval_results = compute_average_results(eval_results)
    save_data_to_json(avg_eval_results, avg_eval_save_path)
    
    print("Evaluation results saved to:", eval_save_path)
    print("Average evaluation results saved to:", avg_eval_save_path)
    print(f"Average nDCG@10: {avg_eval_results.get('ndcg_at_10', 0.0):.4f}")
    print(f"Domain Results: {', '.join([f'{domain}: {result:.4f}' for domain, result in zip(domain_list, domain_results)])}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge search results and evaluate.")
    parser.add_argument("--results_dirs", type=str, nargs="+", required=True, help="List of directories containing search results.")
    parser.add_argument("--weights", type=float, nargs="+", required=True, help="Weights for merging results.")
    parser.add_argument("--norm_weights", action='store_true', help="Whether to normalize weights.")
    parser.add_argument("--merged_model_name", type=str, required=True, help="Name of the merged model.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save merged results and evaluation.")
    parser.add_argument("--ori_split_name", type=str, default="examples", help="Original split name to be replaced in QIDs.")
    parser.add_argument("--top_k", type=int, default=2000, help="Top K results to keep for each query.")
    parser.add_argument("--norm", action='store_true', help="Whether to normalize scores before merging.")
    parser.add_argument("--use_rrf", action='store_true', help="Whether to use RRF for merging results.")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF parameter k for merging results.")
    parser.add_argument("--only_transfer", action='store_true', help="Whether to only transfer results")
    
    args = parser.parse_args()
    
    main(args)
