from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.decoder_only.base import (
    DecoderOnlyEmbedderModelArguments as DynQRIDecoderOnlyEmbedderModelArguments,
)

from arguments import DynQRIDecoderOnlyEmbedderDataArguments, DynQRIDecoderOnlyEmbedderTrainingArguments
from runner import DynQRIEmbedderRunner


def main():
    parser = HfArgumentParser((
        DynQRIDecoderOnlyEmbedderModelArguments,
        DynQRIDecoderOnlyEmbedderDataArguments,
        DynQRIDecoderOnlyEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: DynQRIDecoderOnlyEmbedderModelArguments
    data_args: DynQRIDecoderOnlyEmbedderDataArguments
    training_args: DynQRIDecoderOnlyEmbedderTrainingArguments

    runner = DynQRIEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
