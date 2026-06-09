import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict


class TaskType(Enum):
    # stackexchange
    biology = "Given a Biology post, retrieve relevant passages that help answer the post."
    earth_science = "Given an Earth Science post, retrieve relevant passages that help answer the post."
    economics = "Given an Economics post, retrieve relevant passages that help answer the post."
    psychology = "Given a Psychology post, retrieve relevant passages that help answer the post."
    robotics = "Given a Robotics post, retrieve relevant passages that help answer the post."
    stackoverflow = "Given a Stack Overflow post, retrieve relevant passages that help answer the post."
    sustainable_living = "Given a Sustainable Living post, retrieve relevant passages that help answer the post."
    # coding
    leetcode = "Given a Coding problem, retrieve relevant examples that help answer the problem."
    pony = "Given a Pony question, retrieve relevant passages that help answer the question."
    # theorem-based
    aops = "Given a Math problem, retrieve relevant examples that help answer the problem."
    theoremqa_questions = "Given a Math problem, retrieve relevant examples that help answer the problem. "
    theoremqa_theorems = "Given a Math problem, retrieve relevant theorems that help answer the problem."


class Language(Enum):
    en = 'English'  # 英语


LENGTH_LIST = ["less than 100 words"] * 1 + \
    ["100 to 200 words"] * 2 + \
    ["200 to 300 words"] * 3 + \
    ["300 to 400 words"] * 3 + \
    ["400 to 500 words"] * 2 + \
    ["at least 500 words"] * 1


DIFFICULTY_LIST = ["high school"] * 1 + \
    ["college"] * 2 + \
    ["phd"] * 2


@dataclass
class Task:
    task_type: TaskType
    language: Language
    task_instruction: str = None


def get_task(
    task_type: str,
    language: str,
):
    task_instruction = TaskType[task_type].value.strip()

    task = Task(
        task_type=TaskType[task_type],
        language=Language[language],
        task_instruction=task_instruction,
    )
    return task


def get_generation_prompt(
    task: Task,
    text: str,
) -> str:
    task_to_gen_instruction: Dict[TaskType, str] = {
        # stackexchange
        TaskType.biology: "Given a Biology-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        TaskType.earth_science: "Given an Earch Science-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        TaskType.economics: "Given an Economics-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        TaskType.psychology: "Given a Psychology-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        TaskType.robotics: "Given a Robotics-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        TaskType.stackoverflow: "Given a Coding-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        TaskType.sustainable_living: "Given a Sustainable Living-related passage in {language}, generate a StackExchange post in {language} for which the critical concepts or theories discussed in the passage can serve as references for domain experts to draft an answer.",
        # coding
        TaskType.leetcode: "Given a solved LeetCode problem (with solutions) in {language}, generate a new LeetCode problem in {language} that the underlying algorithms or data structures from the original problem can help solve.",
        TaskType.pony: "Given a Pony documentation passage in {language}, generate a Pony coding instruction in {language} that the Pony syntax described in the passage can help implement.",
        # theorem-based
        TaskType.aops: "Given a Math problem solution in {language}, generate a new Math problem in {language} that the problem-solving skills used in the original problem can help solve.",
        TaskType.theoremqa_questions: "Given a Math problem solution in {language}, generate a new Math problem in {language} that the theorems used in the original problem can help solve.",
        TaskType.theoremqa_theorems: "Given a Math theorem in {language}, generate a Math problem in {language} that the theorem can help solve.",
    }
    
    task_to_gen_output: Dict[TaskType, str] = {
        # stackexchange
        TaskType.biology: "the generated StackExchange post in {language}",
        TaskType.earth_science: "the generated StackExchange post in {language}",
        TaskType.economics: "the generated StackExchange post in {language}",
        TaskType.psychology: "the generated StackExchange post in {language}",
        TaskType.robotics: "the generated StackExchange post in {language}",
        TaskType.stackoverflow: "the generated StackExchange post in {language}",
        TaskType.sustainable_living: "the generated StackExchange post in {language}",
        # coding
        TaskType.leetcode: "the generated LeetCode problem in {language}",
        TaskType.pony: "the generated Pony coding instruction in {language}",
        # theorem-based
        TaskType.aops: "the generated Math problem in {language}",
        TaskType.theoremqa_questions: "the generated Math problem in {language}",
        TaskType.theoremqa_theorems: "the generated Math problem in {language}",
    }
    
    gen_instruction = task_to_gen_instruction[task.task_type]
    gen_output = task_to_gen_output[task.task_type]
    
    gen_instruction = gen_instruction.replace("{language}", task.language.value)
    gen_output = gen_output.replace("{language}", task.language.value)
    
    prefix = "The given content:"
    
    # sample length & difficulty
    length = random.choice(LENGTH_LIST)
    difficulty = random.choice(DIFFICULTY_LIST)
    
    gen_prompt = f"""\
{gen_instruction}

{prefix}
[Begin of Content]
{text}
[End of Content]

Note:
- Your output must always be a string, only containing {gen_output}.
- Your output should be independent of the given content, which means that it should not contain the pronouns such as "it", "this", "that", "the given", "the provided", etc.
- Your output ({gen_output}) should be about {length}.
- Your output ({gen_output}) should require {difficulty} level education to understand.

"""

    gen_prompt += "Remember do not explain your output or output anything else. Your output:"
    
    return gen_prompt


# python string length threshold for document length
DOC_LENGTH_THRESHOLD = (200, 5000)  # (min, max)
