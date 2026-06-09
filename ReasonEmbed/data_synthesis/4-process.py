import os
import json
import argparse
from tqdm import tqdm


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
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    # Theorem-based
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
    
    "general-reasoning": "Given this reasoning-intensive query, find relevant documents that could help answer the question.",
    # mixture-of-thoughts
    "mot_coding": "Given a reasoning-intensive coding question, find relevant documents that could help answer the question.",
    "mot_math": "Given a reasoning-intensive math question, find relevant documents that could help answer the question.",
    "mot_science": "Given a reasoning-intensive science question, find relevant documents that could help answer the question.",
}


# tuned settings for training
SETTING_MAP = {
    "biology": (16, 8),
    "earth_science": (16, 8),
    "economics": (16, 8),
    "psychology": (16, 8),
    "robotics": (16, 8),
    "stackoverflow": (16, 8),
    "sustainable_living": (16, 8),
    
    "leetcode": (2, 64),
    "pony": (64, 2),
    
    "aops": (16, 8),
    "theoremqa_questions": (16, 8),
    "theoremqa_theorems": (16, 8),
}


def main(args):
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file {args.data_file} does not exist.")

    dataset = load_jsonl_data(args.data_file)
    
    updated_dataset = []
    no_pos_or_neg_count = 0
    for example in tqdm(dataset, desc="Processing dataset"):
        docs = example["docs"]
        labels = example["labels"]

        pos, neg = [], []
        for i, label in enumerate(labels):
            if label == -1 or label not in [1, 2, 3, 4, 5]:
                continue
            
            if label >= 3:
                pos.append(docs[i])
            else:
                neg.append(docs[i])
        
        if len(pos) == 0 or len(neg) == 0:
            no_pos_or_neg_count += 1
            continue

        train_group_size, batch_size = SETTING_MAP[args.task_type]
        updated_dataset.append({
            "prompt": BrightShortInstructions[args.task_type],
            "query": example["query"],
            "pos": pos,
            "neg": neg,
            "train_group_size": train_group_size,
            "batch_size": batch_size,
        })

    # min, max, avg
    pos_lengths = [len(example["pos"]) for example in updated_dataset]
    neg_lengths = [len(example["neg"]) for example in updated_dataset]
    
    log_save_dir = os.path.join(os.path.dirname(args.output_file), "logs")
    os.makedirs(log_save_dir, exist_ok=True)
    log_save_path = os.path.join(log_save_dir, os.path.basename(args.output_file).replace('.jsonl', '_log.json'))
    
    log_dict = {
        "task_type": args.task_type,
        "input_file": args.data_file,
        "output_file": args.output_file,
        "sample_count": {
            "total": len(dataset),
            "valid": len(updated_dataset),
            "filtered": no_pos_or_neg_count,
            "filter_ratio": f"{no_pos_or_neg_count / len(dataset):.2%}",
        },
        "num_pos": {
            "min": min(pos_lengths),
            "max": max(pos_lengths),
            "avg": sum(pos_lengths) / len(pos_lengths) if pos_lengths else 0,
        },
        "num_neg": {
            "min": min(neg_lengths),
            "max": max(neg_lengths),
            "avg": sum(neg_lengths) / len(neg_lengths) if neg_lengths else 0,
        },
    }
    with open(log_save_path, 'w') as log_file:
        json.dump(log_dict, log_file, indent=4, ensure_ascii=False)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_data_list_to_jsonl(updated_dataset, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True, help="Task type for the dataset, e.g., 'biology', 'earth_science', etc.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the processed output file")
    args = parser.parse_args()

    main(args)
