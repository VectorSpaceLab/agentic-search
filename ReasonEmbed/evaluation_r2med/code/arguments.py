from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class R2MEDEvalArgs(AbsEvalArgs):
    """
    Argument class for R2MED evaluation.
    """
    use_special_instructions: bool = field(
        default=False, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )
