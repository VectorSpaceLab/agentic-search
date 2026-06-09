import os
import json
import random
import numpy as np
from tqdm import tqdm
from hashlib import md5
from typing import Optional, List
from dataclasses import dataclass, field

import faiss
from transformers import HfArgumentParser
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.inference import AbsEmbedder


BrightShortInstructions = {
    # StackExchange
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the post.",
    # Coding
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question",
    # Theorem-based
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
}


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


@dataclass
class DataArgs:
    """
    Data arguments for candidate mining.
    """
    domain: str = field(
        metadata={"help": "Domain of data."}
    )
    input_file: str = field(
        metadata={"help": "The input file for candidate mining."}
    )
    output_file: str = field(
        metadata={"help": "The output file for candidate mining."}
    )
    candidate_pool: Optional[List[str]] = field(
        default=None, metadata={"help": "The candidate pool for candidate mining. If provided, it should be a jsonl file, each line is a dict with a key 'text'."}
    )
    index_save_dir: Optional[str] = field(
        default=None, metadata={"help": "The path to save the index."}
    )
    search_top_k: int = field(
        default=500, metadata={"help": "The top k for searching."}
    )
    candidates_number: int = field(
        default=15, metadata={"help": "The number of candidates."}
    )
    use_gpu_for_searching: bool = field(
        default=False, metadata={"help": "Whether to use faiss-gpu for searching."}
    )
    search_batch_size: int = field(
        default=64, metadata={"help": "The batch size for searching."}
    )
    add_doc_prefix_for_e5: bool = field(
        default=False, metadata={"help": "Whether to add prefix for e5."}
    )


@dataclass
class ModelArgs:
    """
    Model arguments for embedder.
    """
    embedder_name_or_path: str = field(
        metadata={"help": "The embedder name or path.", "required": True}
    )
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl"]}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    pooling_method: str = field(
        default="cls", metadata={"help": "The pooling method fot the embedder."}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    examples_for_task: Optional[str] = field(
        default=None, metadata={"help": "Examples for task"}
    )
    examples_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format for examples instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    batch_size: int = field(
        default=512, metadata={"help": "Batch size for inference."}
    )
    embedder_query_max_length: int = field(
        default=512, metadata={"help": "Max length for query."}
    )
    embedder_passage_max_length: int = field(
        default=512, metadata={"help": "Max length for passage."}
    )


# create index
def create_index(embeddings: np.ndarray, use_gpu: bool = False):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


# save index
def save_index(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)


# load index
def load_index(index_save_dir: str, use_gpu: bool = False):
    index_path = os.path.join(index_save_dir, "index")
    docid_path = os.path.join(index_save_dir, "docid")
    
    docids = []
    with open(docid_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            docids.append(line.strip())
    
    index = faiss.read_index(index_path)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    return docids, index


# batch search
def batch_search(
    index: faiss.Index,
    query: np.ndarray,
    topk: int = 200,
    batch_size: int = 64
):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


# load corpus dict
def get_corpus_dict(file_path: str):
    corpus_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            if "text" in line.keys():
                if "title" in line.keys():
                    # NOTE: use "{title} {text}" format
                    text = line['title'] + "\n" + line['text']
                else:
                    text = line['text']
                docid = compute_md5(text)
                corpus_dict[docid] = text
            else:
                if "pos" in line.keys():
                    for text in line['pos']:
                        docid = compute_md5(text)
                        corpus_dict[docid] = text
                if "neg" in line.keys():
                    for text in line['neg']:
                        docid = compute_md5(text)
                        corpus_dict[docid] = text
    return corpus_dict


# find knn candidates (core function)
def find_knn_candidates(
    domain: str,
    model: AbsEmbedder,
    input_file: str,
    output_file: str,
    search_top_k: int = 2000,
    candidate_pool: Optional[List[str]] = None,
    index_save_dir: Optional[str] = None,
    candidates_number: int = 15,
    use_gpu: bool = False,
    add_doc_prefix_for_e5: bool = False,
):
    corpus_dict = get_corpus_dict(input_file)
    
    queries = []
    train_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_data.append({
                "prompt": BrightShortInstructions[domain],
                "query": line["query"],
                "docs": [],
                "pos_doc": line["pos"][0]
            })
            queries.append(line['query'])

    if candidate_pool is not None:
        # print(candidate_pool, type(candidate_pool))
        if not isinstance(candidate_pool, list):
            corpus_dict.update(get_corpus_dict(candidate_pool))
        else:
            for cp in candidate_pool:
                corpus_dict.update(get_corpus_dict(cp))

    # encode queries and poses
    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries)

    # encode corpus if necessary
    if index_save_dir is not None:
        if os.path.exists(os.path.join(index_save_dir, 'index')):
            print(f'loading index from {index_save_dir}--------------')
            docids, index = load_index(index_save_dir, use_gpu=use_gpu)
            if len(docids) > len(corpus_dict):
                raise ValueError(f'index size mismatch, loading corpus from {input_file}--------------')
        else:
            print(f'creating index and saving to {index_save_dir}--------------')
            os.makedirs(index_save_dir, exist_ok=True)
            docids = list(corpus_dict.keys())
            
            # sample docids to avoid memory overflow
            if len(docids) > 12_000_000:
                docids = random.sample(docids, 12_000_000)
            
            corpus = [corpus_dict[docid] for docid in docids]
            if add_doc_prefix_for_e5:
                p_vecs = model.encode_corpus(["passage: " + text for text in corpus])
            else:
                p_vecs = model.encode_corpus(corpus)
            index = create_index(p_vecs, use_gpu=False)
            save_index(index, docids, index_save_dir)
            docids, index = load_index(index_save_dir, use_gpu=use_gpu)
    else:
        print(f'creating index without saving--------------')
        docids = list(corpus_dict.keys())
        
        # sample docids to avoid memory overflow
        if len(docids) > 12_000_000:
            docids = random.sample(docids, 12_000_000)
        
        corpus = [corpus_dict[docid] for docid in docids]
        if add_doc_prefix_for_e5:
            p_vecs = model.encode_corpus(["passage: " + text for text in corpus])
        else:
            p_vecs = model.encode_corpus(corpus)
        index = create_index(p_vecs, use_gpu=use_gpu)
    
    results_list = [{} for _ in range(len(queries))]
    
    # search for candidate docs
    print('searching for candidate docs--------------')
    all_scores, all_inxs = batch_search(index, q_vecs, topk=search_top_k)
    assert len(all_inxs) == len(train_data)
    
    for idx, (scores, inxs) in enumerate(zip(all_scores, all_inxs)):
        results = {}
        for score, inx in zip(scores, inxs):
            if inx != -1:
                results[docids[inx]] = score
        results_list[idx].update(results)

    # sort the results
    for idx, results in enumerate(results_list):
        results_list[idx] = sorted(list(results.items()), key=lambda x: x[1], reverse=True)

    # fill candidate docs
    for i, data in enumerate(train_data):
        skip_list = [data['query']]
        
        results = results_list[i]
        candidate_docs = []
        for docid, score in results:
            if docid not in corpus_dict:
                print(f"Skip {docid}")
                continue
            doc = corpus_dict[docid]
            if doc in skip_list:
                continue
            if len(candidate_docs) >= candidates_number:
                break
            candidate_docs.append(docid)
    
        if len(candidate_docs) > candidates_number:
            candidate_docs = random.sample(candidate_docs, candidates_number)
        elif len(candidate_docs) < candidates_number:
            # remove positives from candidates
            docids_rm_pos = list(set(docids))
            candidate_docs.extend(random.sample(docids_rm_pos, candidates_number - len(candidate_docs)))
        
        data['docs'] = [corpus_dict[docid] for docid in candidate_docs]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding="utf-8") as f:
        for data in train_data:
            # remove pos_doc. NOTE (IMPORTANT): exclude source doc here
            pos_doc = data.pop('pos_doc')
            docs_rm_pos = [doc for doc in data['docs'] if doc != pos_doc]
            data['docs'] = docs_rm_pos
            
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Done! The results are saved in {output_file}")


def load_model(model_args: ModelArgs, domain: str):
    model = FlagAutoModel.from_finetuned(
        model_name_or_path=model_args.embedder_name_or_path,
        model_class=model_args.embedder_model_class,
        normalize_embeddings=model_args.normalize_embeddings,
        pooling_method=model_args.pooling_method,
        use_fp16=model_args.use_fp16,
        query_instruction_for_retrieval=BrightShortInstructions[domain],    # domain specific retrieval instruction
        query_instruction_format=model_args.query_instruction_format_for_retrieval,
        devices=model_args.devices,
        examples_for_task=model_args.examples_for_task,
        examples_instruction_format=model_args.examples_instruction_format,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        batch_size=model_args.batch_size,
        query_max_length=model_args.embedder_query_max_length,
        passage_max_length=model_args.embedder_passage_max_length,
    )
    return model


def main(data_args: DataArgs, model_args: ModelArgs):
    domain = data_args.domain
    model = load_model(model_args, domain)

    if os.path.exists(data_args.output_file):
        print(f'{data_args.output_file} already exists, skip')
        return

    find_knn_candidates(
        domain=domain,
        model=model,
        input_file=data_args.input_file,
        output_file=data_args.output_file,
        search_top_k=data_args.search_top_k,
        candidate_pool=data_args.candidate_pool,
        index_save_dir=data_args.index_save_dir,
        candidates_number=data_args.candidates_number,
        use_gpu=data_args.use_gpu_for_searching,
        add_doc_prefix_for_e5=data_args.add_doc_prefix_for_e5,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((
        DataArgs,
        ModelArgs
    ))
    data_args, model_args = parser.parse_args_into_dataclasses()
    data_args: DataArgs
    model_args: ModelArgs
    main(data_args, model_args)
