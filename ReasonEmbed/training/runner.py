import logging
from typing import Tuple

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from FlagEmbedding.finetune.embedder.decoder_only.base import (
    DecoderOnlyEmbedderRunner,
    DecoderOnlyEmbedderModelArguments as DynQRIDecoderOnlyEmbedderModelArguments
)
from FlagEmbedding.finetune.embedder.decoder_only.base.load_model import get_model
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel, EmbedderTrainerCallbackForDataRefresh

from arguments import (
    DynQRIDecoderOnlyEmbedderDataArguments,
    DynQRIDecoderOnlyEmbedderTrainingArguments
)
from modeling import DynQRIBiDecoderOnlyEmbedderModel
from trainer import DynQRIDecoderOnlyEmbedderTrainer
from custom_dataset import DynQRIEmbedderSameDatasetTrainDataset, DynQRIEmbedderSameDatasetCollator

logger = logging.getLogger(__name__)


class DynQRIEmbedderRunner(DecoderOnlyEmbedderRunner):
    """Runner class for dynamic QRI training.

    Args:
        model_args (DynQRIDecoderOnlyEmbedderModelArguments): Model arguments instance.
        data_args (DynQRIDecoderOnlyEmbedderDataArguments): Data arguments instance.
        training_args (DynQRIDecoderOnlyEmbedderTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: DynQRIDecoderOnlyEmbedderModelArguments,
        data_args: DynQRIDecoderOnlyEmbedderDataArguments,
        training_args: DynQRIDecoderOnlyEmbedderTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: DynQRIDecoderOnlyEmbedderModelArguments
        self.data_args: DynQRIDecoderOnlyEmbedderDataArguments
        self.training_args: DynQRIDecoderOnlyEmbedderTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            add_eos_token=True,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.model_args.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.model_args.additional_special_tokens} already exists in the tokenizer.")
        base_model = get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer))

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        qri_score_kwargs = {
            "qri_start_step": self.training_args.qri_start_step,
            "qri_score_mapping": self.training_args.qri_score_mapping,
            "qri_score_clamp_min": self.training_args.qri_score_clamp_min,
            "qri_score_clamp_max": self.training_args.qri_score_clamp_max,
            "weights_norm": self.training_args.weights_norm
        }

        model = DynQRIBiDecoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            **qri_score_kwargs
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_train_dataset(self) -> DynQRIEmbedderSameDatasetTrainDataset:
        if self.data_args.same_dataset_within_batch:
            train_dataset = DynQRIEmbedderSameDatasetTrainDataset(
                args=self.data_args,
                default_batch_size=self.training_args.per_device_train_batch_size,
                seed=self.training_args.seed,
                tokenizer=self.tokenizer,
                process_index=self.training_args.process_index,
                num_processes=self.training_args.world_size
            )
            self.training_args.per_device_train_batch_size = 1
            self.training_args.dataloader_num_workers = 0   # avoid multi-processing
        else:
            raise NotImplementedError("Only support same_dataset_within_batch=True for DynQRI training.")
        return train_dataset

    def load_data_collator(self) -> DynQRIEmbedderSameDatasetCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
        """
        if self.data_args.same_dataset_within_batch:
            EmbedCollator = DynQRIEmbedderSameDatasetCollator
        else:
            raise NotImplementedError("Only support same_dataset_within_batch=True for DynQRI training.")

        data_collator = EmbedCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            reasoning_query_max_len=self.data_args.reasoning_query_max_len if self.data_args.reasoning_query_max_len is not None else self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            sub_batch_size=self.training_args.sub_batch_size,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator

    def load_trainer(self) -> DynQRIDecoderOnlyEmbedderTrainer:
        """Load the trainer.

        Returns:
            DynQRIDecoderOnlyEmbedderTrainer: Loaded trainer instance.
        """
        trainer = DynQRIDecoderOnlyEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer
