from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainingArguments, AbsEmbedderDataArguments
)


@dataclass
class DynQRIDecoderOnlyEmbedderDataArguments(AbsEmbedderDataArguments):
    reasoning_query_max_len: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization for reasoning query. Sequences longer than this will be truncated. Default to `query_max_len` if not set."
        },
    )


@dataclass
class DynQRIDecoderOnlyEmbedderTrainingArguments(AbsEmbedderTrainingArguments):
    qri_start_step: int = field(default=-1, metadata={"help": "Number of steps to start dynamic QRI"})
    
    qri_score_mapping: str = field(
        default="clamp",
        metadata={
            "help": "Mapping function to convert QRI scores to weights before normalizing. Options: \
                'clamp' (clamp scores to [qri_score_clamp_min, qri_score_clamp_max]), \
                'none' (no mapping, use raw scores) \
                "
        },
    )
    qri_score_clamp_min: float = field(default=0.0, metadata={"help": "Minimum clamp value for QRI scores when using 'clamp' mapping"})
    qri_score_clamp_max: float = field(default=5.0, metadata={"help": "Maximum clamp value for QRI scores when using 'clamp' mapping"})
    
    weights_norm: str = field(
        default="l1",
        metadata={
            "help": "Normalization method for QRI weights. Options: 'l1' (weights sum to 1), 'none' (no normalization)"
        },
    )
