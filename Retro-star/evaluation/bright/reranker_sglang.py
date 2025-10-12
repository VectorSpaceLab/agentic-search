import re
import time
import logging
import sglang as sgl

from tqdm import tqdm
from typing import Optional, List, Tuple


logger = logging.getLogger(__name__)


class SGLangReasoningLLMReranker:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 8192,
        tp_size: int = 1,
        dp_size: int = 1,
        context_length: int = 32768,
        random_seed: int = 42,
        batch_size: int = 128,
        relevance_definition: Optional[str] = None,
        query_type: Optional[str] = "query",
        doc_type: Optional[str] = "document",
        truncate_query_max_length: Optional[int] = None,
        truncate_doc_max_length: Optional[int] = None,
        sample_k: int = 1,
        enable_thinking: bool = False,
        enable_full_parallelism: bool = False,
    ):
        self.reranker_model_class = "sglang-reasoning"
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.context_length = context_length
        self.random_seed = random_seed

        self.truncate_query_max_length = truncate_query_max_length
        self.truncate_doc_max_length = truncate_doc_max_length
        self.max_prompt_length = context_length - max_new_tokens - 128
        self.sample_k = sample_k
        self.enable_thinking = enable_thinking
        self.enable_full_parallelism = enable_full_parallelism

        if relevance_definition is None:
            relevance_definition = f"Given a query ({query_type}) and a document ({doc_type}), the document is relevant to the query if the document can help answer the query."
        
        self.relevance_definition = relevance_definition
        self.query_type = query_type
        self.doc_type = doc_type

        self.score_pattern = r"<score>\s*(\d+)\s*</score>"

        self.prompt_template = self.get_prompt_template()
        print("Prompt Template:\n", self.prompt_template)
        self.init_flag = False

    def __del__(self):
        if self.init_flag:
            self.model.shutdown()

    def _init_engine(self):
        if not self.init_flag:
            self.model = sgl.Engine(
                model_path=self.model_name_or_path,
                context_length=self.context_length,
                tp_size=self.tp_size,
                dp_size=self.dp_size,
                random_seed=self.random_seed,
            )
            self.tokenizer = self.model.tokenizer_manager.tokenizer
            self.tokenizer.padding_side = "left"
            self.init_flag = True

    def _truncate_texts(self, texts: list, max_length: Optional[int] = None):
        if max_length is None:
            max_length = self.max_prompt_length
        token_ids = self.tokenizer(texts)["input_ids"]
        truncated_token_ids = [ids[:max_length] for ids in token_ids]
        return self.tokenizer.batch_decode(truncated_token_ids)

    def compute_score(
        self,
        sentence_pairs: List[Tuple[str, str]],
    ):
        self._init_engine()
        print("Relevance Definition:\n", self.relevance_definition)

        sampling_params = {
            "n": self.sample_k,
            "temperature": 0.6,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "max_new_tokens": self.max_new_tokens,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }

        if self.enable_full_parallelism:
            sampling_params["n"] = 1

        scores = []
        running_time, completion_tokens = 0.0, 0.0
        num_batches = (len(sentence_pairs) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(sentence_pairs), self.batch_size), desc="Reasoning Reranking", total=num_batches):
            batch = sentence_pairs[i:i + self.batch_size]
            
            batch_queries, batch_docs = [], []
            for pair in batch:
                query, doc = pair
                batch_queries.append(query)
                batch_docs.append(doc)

            batch_queries = self._truncate_texts(batch_queries, self.truncate_query_max_length)
            batch_docs = self._truncate_texts(batch_docs, self.truncate_doc_max_length)

            prompts = [
                self.prompt_template.format(
                    relevance_definition=self.relevance_definition,
                    query=query,
                    doc=doc,
                    query_type=self.query_type,
                    doc_type=self.doc_type,
                )
                for query, doc in zip(batch_queries, batch_docs)
            ]

            truncated_prompts_token_ids = []
            prompts_token_ids = self.tokenizer(prompts)["input_ids"]
            for prompt_token_ids in prompts_token_ids:
                if len(prompt_token_ids) > self.max_prompt_length:
                    print(f"Truncating prompt from {len(prompt_token_ids)} tokens to {self.max_prompt_length} tokens.")
                    prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
                truncated_prompts_token_ids.append(prompt_token_ids)
            truncated_prompts = self.tokenizer.batch_decode(truncated_prompts_token_ids)

            messages = [[{"role": "user", "content": prompt}] for prompt in truncated_prompts]
            input_texts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            if self.enable_full_parallelism:
                input_texts = [text for text in input_texts for _ in range(self.sample_k)]

            time_start = time.time()
            outputs = self.model.generate(
                input_texts,
                sampling_params=sampling_params,
            )
            running_time += time.time() - time_start

            assert len(outputs) == len(prompts) * self.sample_k, f"Expected {len(prompts) * self.sample_k} outputs, but got {len(outputs)} outputs."
            for i in range(0, len(outputs), self.sample_k):
                k_outputs = outputs[i:i + self.sample_k]
                score = 0
                for output in k_outputs:
                    completion_tokens += output["meta_info"]["completion_tokens"]
                    try:
                        score += int(re.search(self.score_pattern, output["text"]).group(1))
                    except Exception:
                        score += 0
                score = score / self.sample_k
                scores.append(score)

        completion_tokens = completion_tokens / (len(sentence_pairs) * self.sample_k)

        return scores, running_time, completion_tokens

    def get_prompt_template(self):
        prompt_template = """\
Here is the **relevance definition** in a retrieval task: {relevance_definition}

Now given a **query** ({query_type}) and a **document** ({doc_type}) in this retrieval task, your mission is to perform the following steps.

1. Query Analysis: Think to reason and describe what information would be most helpful in answering the query.
2. Document Analysis: Discuss how the information provided by the document fulfills or fails to fulfill the requirements implied by the query.
3. Relevance Annotation: Based on the relevance definition and the insights from the previous two steps, clearly justify your final relevance annotation result and annotate an integer score from a scale of 0 to 100. Please use the following guide:
    - **80-100 (Highly Relevant):** The document directly and comprehensively addresses the query's intent. It is a core and authoritative answer.
    - **60-80 (Relevant):** The document substantially addresses the query's intent, providing most of the key information, but might miss some minor details.
    - **40-60 (Moderately Relevant):** The document is on-topic and addresses a part of the query's intent, but it is not a comprehensive answer.
    - **20-40 (Slightly Relevant):** The document mentions keywords from the query, but its main topic is different. It offers very limited value.
    - **0-20 (Irrelevant):** The document does not address the query's intent at all and is off-topic.

After providing your detailed analysis and justification for all the steps above, conclude your entire response with the final relevance score. The score must be placed strictly between the <score> tags. There should be no other text or explanation inside the tags:
<score>
[From a scale of 0 to 100, annotate the degree of relevance between the query and the document.]
</score>

Query ({query_type}):
[Begin of Query]
{query}
[End of Query]

Document ({doc_type}):
[Begin of Document]
{doc}
[End of Document]
"""
        return prompt_template
