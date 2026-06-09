import math
import random
from dataclasses import dataclass
from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderSameDatasetTrainDataset, AbsEmbedderSameDatasetCollator
)


class DynQRIEmbedderSameDatasetTrainDataset(AbsEmbedderSameDatasetTrainDataset):
    """Dataset for dynamic QRI training."""
    def __getitem__(self, _):
        batch_indices, no_in_batch_neg_flag = self.batch_datas[self.step]    # extend here
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, reasoning_queries, passages, teacher_scores = self._create_batch_data(batch_raw_data=batch_data)
        return queries, reasoning_queries, passages, teacher_scores, no_in_batch_neg_flag
    
    def _create_batch_data(self, batch_raw_data):
        """Create a comple batch of data with queries, reasoning_queries, documents and teacher scores.

        Args:
            batch_raw_data (datasets.Dataset): One batch of raw data.

        Returns:
            List[str]: Queries with instruction format.
            List[str]: Reasoning queries with instruction format.
            List[str]: Documents with instruction format.
            List[float]: Teacher scores for model distillation.
        """
        queries, reasoning_queries, passages, teacher_scores = [], [], [], []

        train_group_size, data_type = self._get_train_group_size(batch_raw_data)

        for i in range(len(batch_raw_data['query'])):
            if data_type is not None:
                assert batch_raw_data['type'][i] == data_type, f"Data type is not consistent in the same batch"

            queries.append(
                self.args.query_instruction_format.format(
                    batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                    batch_raw_data['query'][i]
                )
            )
            reasoning_queries.append(
                self.args.query_instruction_format.format(
                    batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                    batch_raw_data['reasoning_query'][i]
                )
            )
            
            tmp_passages = []
            pos_idx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            pos = self._shuffle_text(batch_raw_data['pos'][i][pos_idx])
            tmp_passages.append(pos)

            neg_all_idx = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
            for neg_idx in neg_idxs:
                tmp_passages.append(batch_raw_data['neg'][i][neg_idx])

            if self.args.knowledge_distillation:
                if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                    teacher_scores.append(batch_raw_data['pos_scores'][i][pos_idx])
                for neg_idx in neg_idxs:
                    if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                        teacher_scores.append(batch_raw_data['neg_scores'][i][neg_idx])
            else:
                teacher_scores = None

            if data_type is not None and data_type in ['symmetric_sts', 'symmetric_clustering']:
                tmp_passages = [
                    self.args.query_instruction_format.format(
                        batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                        p
                    ) for p in tmp_passages
                ]
            else:
                if self.args.passage_instruction_for_retrieval is not None:
                    tmp_passages = [
                        self.args.passage_instruction_format.format(
                            self.args.passage_instruction_for_retrieval, p
                        ) for p in tmp_passages
                    ]

            passages.extend(tmp_passages)

            if teacher_scores is not None:
                if len(teacher_scores) > 0 and len(passages) > 0:
                    assert len(teacher_scores) == len(passages)

        return queries, reasoning_queries, passages, teacher_scores


@dataclass
class DynQRIEmbedderSameDatasetCollator(AbsEmbedderSameDatasetCollator):
    """Data collator for dynamic QRI training."""
    reasoning_query_max_len: int = 32
    
    def __call__(self, features):
        queries = features[0][0]
        reasoning_queries = features[0][1]
        passages = features[0][2]
        teacher_scores = features[0][3]
        no_in_batch_neg_flag = features[0][4]

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        reasoning_queries_inputs = self.tokenizer(
            reasoning_queries,
            truncation=True,
            max_length=self.reasoning_query_max_len,
            return_tensors=None
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            
            reasoning_q_collated = self.tokenizer.pad(
                reasoning_queries_inputs,
                padding=self.padding,
                max_length=self.reasoning_query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            reasoning_q_collated = []
            for i in range(0, len(reasoning_queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(reasoning_queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in reasoning_queries_inputs.items():
                    sub_features[k] = v[start:end]
                reasoning_q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.reasoning_query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "reasoning_queries": reasoning_q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }
