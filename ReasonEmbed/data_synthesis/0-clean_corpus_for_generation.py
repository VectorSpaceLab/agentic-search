import os
import json
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from llm import LLM


BRIGHT_DATA_ROOT = os.environ.get("BRIGHT_DATA_ROOT", "./bright_short/data")
DOMAIN_MAP_DICT = {
    "biology": "Biology",
    "earth_science": "Earth Science",
    "economics": "Economics",
    "psychology": "Psychology",
    "robotics": "Robotics",
    "stackoverflow": "Coding",
    "sustainable_living": "Sustainable Living",
    "leetcode": "Coding",
    "pony": "Coding",
    "aops": "Math",
    "theoremqa_questions": "Math",
    "theoremqa_theorems": "Math",
}


def get_annotation_prompt(data: dict, domain: str):
    text = data["text"]
    
    prompt = f"""\
Given a passage, determine whether it belongs to the domain: {DOMAIN_MAP_DICT[domain]}.

The given passage:
[Begin of Passage]
{text}
[End of Passage]

Note:
- Your output must always be "Yes" or "No".

Remember do not explain your output or output anything else. Your output:
"""
    return prompt


def load_llm():
    annotator = LLM(
        model=os.environ.get("GENERATION_MODEL", "Qwen2-5-72B-Instruct"),
        model_type=os.environ.get("GENERATION_MODEL_TYPE", "open-source"),
        port=int(os.environ.get("GENERATION_PORT", "8000")),
    )
    return annotator


def load_corpus_data(path: str):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            data_list.append(data)
    return data_list


def annotate_data_list(annotator: LLM, data_list: list, domain: str, thread_count: int = 64):
    def annotate_single(data: dict):
        prompt = get_annotation_prompt(data, domain)
        response = annotator.chat(prompt)
        try:
            result = response[0]
            if result == "Yes":
                label = 1
            elif result == "No":
                label = 0
            else:
                label = -1
        except:
            label = -1
        return label
    
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        results = list(tqdm(executor.map(
            annotate_single,
            data_list
        ), total=len(data_list), desc="Annotating"))
    
    new_data_dict = defaultdict(list)
    for label, data in zip(results, data_list):
        if label == 1:
            new_data_dict["true"].append(data)
        elif label == 0:
            new_data_dict["false"].append(data)
        elif label == -1:
            new_data_dict["nan"].append(data)
        else:
            raise ValueError(f"Unknown label: {label}")
    
    return new_data_dict    


def save_new_data_dict(new_data_dict: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    stat_dict = {
        "total": 0,
    }
    
    for label_str, data_list in new_data_dict.items():
        stat_dict[label_str] = len(data_list)
        stat_dict["total"] += len(data_list)
        
        save_path = os.path.join(save_dir, f"corpus_annotated_{label_str}.jsonl")
        with open(save_path, "w", encoding="utf-8") as f:
            for data in tqdm(data_list, desc="Saving data"):
                f.write(json.dumps(data) + "\n")
    
    stat_save_path = os.path.join(save_dir, "stat.json")
    with open(stat_save_path, "w", encoding="utf-8") as f:
        json.dump(stat_dict, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully saved to {save_dir}")


def main():
    annotator = load_llm()
    thread_count = 64
    
    domain_list = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "leetcode", "pony", "aops", "theoremqa_questions", "theoremqa_theorems"]
    
    for domain in domain_list:
        ori_corpus_data_path = os.path.join(BRIGHT_DATA_ROOT, domain, "corpus.jsonl")
        
        data_list = load_corpus_data(ori_corpus_data_path)
        
        new_data_dict = annotate_data_list(annotator, data_list, domain, thread_count=thread_count)
        
        save_dir = os.path.join(BRIGHT_DATA_ROOT, domain, "annotated_for_generation")
        save_new_data_dict(new_data_dict, save_dir)
    
    print("All done!")


if __name__ == "__main__":
    main()
