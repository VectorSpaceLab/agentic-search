from dataclasses import dataclass, field
from typing import Optional, List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs, AbsEvalModelArgs


@dataclass
class BrightEvalArgs(AbsEvalArgs):
    """
    Argument class for Bright evaluation.
    """
    task_type: str = field(
        default="short", metadata={"help": "The task type to evaluate on. Available options: ['short']. Default: short", "choices": ["short"]}
    )

    multiple_pass_reranking_paths: List[str] = field(
        default_factory=lambda: ["100@1"],
        metadata={"help": "The paths for multiple pass reranking. Default: ['100@1']"}
    )

    reranker_scores_cache_dir: str = field(
        default="bright-evaluation/.reranker_scores_cache",
        metadata={"help": "The directory to cache reranker scores."},
    )

    reranker_enable_cache: bool = field(
        default=True,
        metadata={"help": "Whether to enable caching of the reranker scores. Default: True"}
    )

    reranker_overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the reranker scores cache. Default: False"}
    )

@dataclass
class BrightEvalModelArgs(AbsEvalModelArgs):
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl', 'custom']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl", "custom"]}
    )

    reranker_model_class: Optional[str] = field(
        default=None, metadata={"help": "The reranker model class. Available classes: ['encoder-only-base', 'decoder-only-base', 'decoder-only-layerwise', 'decoder-only-lightweight', 'sglang-reasoning']. Default: None. For the custom model, you need to specify the model class.", "choices": ["encoder-only-base", "decoder-only-base", "decoder-only-layerwise", "decoder-only-lightweight", "sglang-reasoning"]}
    )

    reranker_sglang_context_length: int = field(
        default=32768,
        metadata={"help": "The context length."}
    )
    
    reranker_sglang_tp_size: int = field(
        default=1,
        metadata={"help": "The tensor parallel size."}
    )

    reranker_sglang_dp_size: int = field(
        default=1,
        metadata={"help": "The data parallel size."}
    )

    reranker_sglang_seed: int = field(
        default=42,
        metadata={"help": "The random seed."}
    )

    reranker_max_new_tokens: int = field(
        default=8192,
        metadata={"help": "The max new tokens."}
    )

    reranker_truncate_query_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length to truncate the query. If None, no truncation is applied."}
    )

    reranker_truncate_doc_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length to truncate the document. If None, no truncation is applied."}
    )

    reranker_sample_k: int = field(
        default=1,
        metadata={"help": "The number of samples to generate for reranking. Default: 1"}
    )

    reranker_enable_full_parallelism: bool = field(
        default=False,
        metadata={"help": "Whether to enable full parallelism for reranking. Default: False"}
    )

    reranker_enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable thinking for reranking. Default: True"}
    )
