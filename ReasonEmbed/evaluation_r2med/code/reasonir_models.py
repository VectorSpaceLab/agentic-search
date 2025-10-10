import os
from sentence_transformers import SentenceTransformer
from typing import Optional


# MultiGPUModel: from https://github.com/embeddings-benchmark/mteb/issues/27
class MultiGPUModel:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.gpu_pool = self.model.start_multi_process_pool()

    def encode(self, sentences, **kwargs):
        return self.model.encode_multi_process(sentences, self.gpu_pool, **kwargs)


class ReasonIREncoder:
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        normalize_embeddings: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        batch_size: int = 256,
        corpus_batch_size: int = 0,
        **kwargs
    ):
        self.name = os.path.basename(model_name_or_path)

        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)
        self.model = MultiGPUModel(self.model)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.corpus_batch_size = (
            corpus_batch_size if corpus_batch_size > 0 else batch_size
        )

    def encode_corpus(self, corpus, **kwargs):
        input_texts = corpus
        if isinstance(corpus[0], dict):
            input_texts = [
                "{} {}".format(doc.get("title", ""), doc.get("text", "")).strip()
                for doc in corpus
            ]
        self.model.model.max_seq_length = self.passage_max_length
        return self.model.encode(
            input_texts,
            prompt="<|embed|>\n",
            batch_size=self.corpus_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

    def encode_queries(self, queries, **kwargs):
        input_texts = queries
        if isinstance(queries[0], dict):
            input_texts = [doc.get("text", "").strip() for doc in queries]
        self.model.model.max_seq_length = self.query_max_length
        return self.model.encode(
            input_texts,
            prompt="<|user|>\n{}\n<|embed|>\n".format(self.query_instruction_for_retrieval),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

    def __str__(self):
        return self.name
