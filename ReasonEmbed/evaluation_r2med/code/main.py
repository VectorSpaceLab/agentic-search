from transformers import HfArgumentParser

from FlagEmbedding.abc.evaluation import (
    AbsEvalModelArgs as R2MEDEvalModelArgs,
)

from arguments import R2MEDEvalArgs
from runner import R2MEDEvalRunner


def main():
    parser = HfArgumentParser((
        R2MEDEvalArgs,
        R2MEDEvalModelArgs
    ))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: R2MEDEvalArgs
    model_args: R2MEDEvalModelArgs

    runner = R2MEDEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
