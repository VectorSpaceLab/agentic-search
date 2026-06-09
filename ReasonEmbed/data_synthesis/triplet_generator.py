import os
import json
import random
from tqdm import tqdm
from hashlib import md5
from warnings import warn
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from llm import LLM
from utils import clean_content
from constant import Task, get_task, get_generation_prompt


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


class TripletGenerator(LLM):
    def __init__(
        self,
        model: str = "Qwen2-5-72B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
        cache_dir: Optional[str] = None
    ):
        super().__init__(model, model_type, port)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def generate_triplets(
        self,
        data: dict,
        task: Task,
        debug_mode: bool = False,
        **kwargs
    ):
        result_list = []

        try:
            text = data["text"]
            
            gen_prompt = get_generation_prompt(
                task=task,
                text=text,
            )
            response = self.chat(gen_prompt, **kwargs)[0]
            
            query = clean_content(response)
            pos = text
                
            if debug_mode:
                result = {
                    "generation_prompt": gen_prompt,
                    "prompt": task.task_instruction,
                    "query": query,
                    "pos": [pos],
                    "neg": [],
                    "response": response
                }
            else:
                result = {
                    "prompt": task.task_instruction,
                    "query": query,
                    "pos": [pos],
                    "neg": []
                }
            
            result_list.append(result)
        except Exception as e:
            warn(f"Error: {e}")
        
        return result_list

    def run_single(
        self,
        data: dict,
        task: Task,
        debug_mode: bool = False,
        **kwargs
    ):
        result_list = []

        docid = compute_md5(data["text"])
        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            if os.path.exists(gen_data_cache_path):
                try:
                    with open(gen_data_cache_path, "r", encoding="utf-8") as f:
                        result_list = json.load(f)
                except:
                    print(f"load error: {gen_data_cache_path}")
                
                if len(result_list) > 0:
                    return result_list

        triplets = self.generate_triplets(
            data,
            task=task,
            debug_mode=debug_mode,
            **kwargs
        )
        if len(triplets) == 0:
            return result_list
        
        result = triplets[0]
        if debug_mode:
            result["docid"] = docid
        
        result_list.append(result)
        
        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            with open(gen_data_cache_path, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)
        
        return result_list

    def run(
        self,
        positives: List[dict],
        task_type: str,
        language: str = "en",
        tqdm_desc: str = "Generating triplets",
        debug_mode: bool = False,
        thread_count: int = 1,
        **kwargs
    ):
        task = get_task(
            task_type=task_type,
            language=language,
        )
        
        result_list = []

        def process_positive(positive):
            return self.run_single(
                data=positive,
                task=task,
                debug_mode=debug_mode,
                **kwargs
            )
        # Use thread pool for parallel processing with tqdm progress bar.
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm(executor.map(
                process_positive,
                positives
            ), total=len(positives), desc=tqdm_desc))

        # Collect results into result_list.
        for res in results:
            if isinstance(res, list):
                result_list.extend(res)
            else:
                result_list.append(res)

        return result_list
