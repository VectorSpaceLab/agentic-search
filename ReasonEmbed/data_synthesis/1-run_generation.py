import os
import json
import time
import argparse
import random
from hashlib import md5
import multiprocessing as mp
from typing import List, Optional

from constant import TaskType, Language
from corpus_generator import CorpusGenerator
from triplet_generator import TripletGenerator


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        help='The task type to generate data for',
        choices=[t.name for t in TaskType]
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='The path to save the generated data'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='The cache directory'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='The language to generate for. ISO 639-1 code. Default: en',
        choices=[l.name for l in Language]
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='The number of examples to use for generation. Default: -1. Use all available examples.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen2.5-72B-Instruct',
        help='The model to use for generation. Default: Qwen2.5-72B-Instruct'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='open-source',
        help='The type of model to use for generation. Default: open-source',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='The port for vllm.'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=1,
        help='The number of processes to use for generation. Default: 1'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The random seed for reproducibility. Default: 42'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite the existing data.'
    )
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        help='Whether to open debug mode.'
    )
    parser.add_argument(
        '--use_cleaned_corpus',
        action='store_true',
        help='Whether to use the cleaned corpus for generation. Default: False'
    )
    args = parser.parse_args()
    return args


def gen_triplets(
    model: str,
    model_type: str,
    port: int,
    positives: List[dict],
    task_type: str,
    language: str,
    tqdm_desc: str = "Generating triplets",
    thread_count: int = 1,
    gen_cache_dir: Optional[str] = None,
    debug_mode: bool = False,
):
    triplet_generator = TripletGenerator(model, model_type, port, cache_dir=gen_cache_dir)
    triplets = triplet_generator.run(
        positives=positives,
        task_type=task_type,
        language=language,
        tqdm_desc=tqdm_desc,
        debug_mode=debug_mode,
        thread_count=thread_count,
    )
    return triplets


def get_save_path(
    save_dir: str,
    task_type: str,
    language: str,
):
    save_dir = os.path.join(save_dir, language, task_type)
    file_name = f"{language}-triplets.jsonl"
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_path


def save_triplets(
    triplets: list,
    save_dir: str,
    task_type: str,
    language: str,
):
    if len(triplets) == 0:
        print(f"No triplets to save: {task_type} | {language}")
        return
    
    save_path = get_save_path(save_dir, task_type, language)
    query_md5s = set()
    pos_md5s = set()
    old_triplets = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                triplet = json.loads(line)
                old_triplets.append(triplet)
                query_md5s.add(compute_md5(triplet['query']))
                pos_md5s.add(compute_md5(triplet['pos'][0]))

    with open(save_path, 'w', encoding='utf-8') as f:
        for triplet in old_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        
        for triplet in triplets:
            _query_md5 = compute_md5(triplet['query'])
            _pos_md5 = compute_md5(triplet['pos'][0])
            if _query_md5 in query_md5s or _pos_md5 in pos_md5s:
                continue
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
    print(f"Triplets saved to {save_path}")


def main(args):
    model = args.model
    model_type = args.model_type
    port = args.port

    num_samples = args.num_samples
    seed = args.seed
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    task_type = args.task_type
    language = args.language

    save_dir = args.save_dir
    cache_dir = args.cache_dir
    num_processes = min(args.num_processes, int(mp.cpu_count() * 0.8))
    overwrite = args.overwrite
    debug_mode = args.debug_mode
    use_cleaned_corpus = args.use_cleaned_corpus
    
    save_path = get_save_path(save_dir, task_type, language)
    if os.path.exists(save_path) and not overwrite:
        data = []
        with open(save_path) as f:
            for line in f:
                data.append(json.loads(line))
        if len(data) >= num_samples * 0.8:
            print(f"Triplets already exist at {save_path}. Skipping generation.")
            return
        else:
            print(f"Triplets already exist at {save_path}. But samples is really small, continue generation.")
            num_samples = int((num_samples - len(data)) * 1.25)  # consider the filtered samples

    corpus_generator = CorpusGenerator(cache_dir)

    positives = corpus_generator.run(
        task_type=task_type,
        num_samples=num_samples,
        use_cleaned_corpus=use_cleaned_corpus,
    )

    print("=================== Generate training data ===================")
    print(f'Task Type: {task_type} | Language: {language}')
    start_time = time.time()
    triplets = gen_triplets(
        model=model,
        model_type=model_type,
        port=port,
        positives=positives,
        task_type=task_type,
        language=language,
        thread_count=num_processes,
        gen_cache_dir=os.path.join(save_dir, language, task_type, "gen_cache_dir"),
        debug_mode=debug_mode,
    )
    save_triplets(
        triplets=triplets,
        save_dir=save_dir,
        task_type=task_type,
        language=language,
    )
    end_time = time.time()
    print("=============================================================")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("=============================================================")
    print("DONE!")


if __name__ == "__main__":
    args = get_args()
    main(args)
