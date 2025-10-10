import logging
from FlagEmbedding.abc.evaluation import AbsEvalRunner
from FlagEmbedding import FlagAutoModel, FlagAutoReranker

from data_loader import R2MEDEvalDataLoader
from prompts import R2MEDInstructions
from reasonir_models import ReasonIREncoder
from searcher import ReasonIRRetriever

logger = logging.getLogger(__name__)


class R2MEDEvalRunner(AbsEvalRunner):
    def load_data_loader(self) -> R2MEDEvalDataLoader:
        data_loader = R2MEDEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

    @staticmethod
    def get_models(model_args):
        """Get the embedding and reranker model

        Args:
            model_args (AbsEvalModelArgs): :class:AbsEvalModelArgs object with the model arguments.

        Returns:
            Tuple[AbsEmbedder, Union[AbsReranker, None]]: A :class:AbsEmbedder object of embedding model, and 
                :class:AbsReranker object of reranker model if path provided.
        """
        if "ReasonIR-8B" in model_args.embedder_name_or_path:
            embedder = ReasonIREncoder(
                model_name_or_path=model_args.embedder_name_or_path,
                normalize_embeddings=model_args.normalize_embeddings,
                query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
                trust_remote_code=model_args.trust_remote_code,
                batch_size=model_args.embedder_batch_size,
                query_max_length=model_args.embedder_query_max_length,
                passage_max_length=model_args.embedder_passage_max_length,
                model_kwargs={"torch_dtype": "auto"},
            )
        else:
            embedder = FlagAutoModel.from_finetuned(
                model_name_or_path=model_args.embedder_name_or_path,
                model_class=model_args.embedder_model_class,
                normalize_embeddings=model_args.normalize_embeddings,
                pooling_method=model_args.pooling_method,
                use_fp16=model_args.use_fp16,
                query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
                query_instruction_format=model_args.query_instruction_format_for_retrieval,
                devices=model_args.devices,
                examples_for_task=model_args.examples_for_task,
                examples_instruction_format=model_args.examples_instruction_format,
                trust_remote_code=model_args.trust_remote_code,
                cache_dir=model_args.cache_dir,
                batch_size=model_args.embedder_batch_size,
                query_max_length=model_args.embedder_query_max_length,
                passage_max_length=model_args.embedder_passage_max_length,
            )
            embedder.model.config._name_or_path = model_args.embedder_name_or_path
        reranker = None
        if model_args.reranker_name_or_path is not None:
            reranker = FlagAutoReranker.from_finetuned(
                model_name_or_path=model_args.reranker_name_or_path,
                model_class=model_args.reranker_model_class,
                peft_path=model_args.reranker_peft_path,
                use_fp16=model_args.use_fp16,
                use_bf16=model_args.use_bf16,
                query_instruction_for_rerank=model_args.query_instruction_for_rerank,
                query_instruction_format=model_args.query_instruction_format_for_rerank,
                passage_instruction_for_rerank=model_args.passage_instruction_for_rerank,
                passage_instruction_format=model_args.passage_instruction_format_for_rerank,
                cache_dir=model_args.cache_dir,
                trust_remote_code=model_args.trust_remote_code,
                devices=model_args.devices,
                normalize=model_args.normalize,
                prompt=model_args.prompt,
                cutoff_layers=model_args.cutoff_layers,
                compress_layers=model_args.compress_layers,
                compress_ratio=model_args.compress_ratio,
                batch_size=model_args.reranker_batch_size,
                query_max_length=model_args.reranker_query_max_length,
                max_length=model_args.reranker_max_length,
            )
            reranker.model.config._name_or_path = model_args.reranker_name_or_path
        return embedder, reranker

    def load_retriever_and_reranker(self):
        """Load retriever and reranker for evaluation

        Returns:
            Tuple[EvalDenseRetriever, Union[EvalReranker, None]]: A :class:EvalDenseRetriever object for retrieval, and a
                :class:EvalReranker object if reranker provided.
        """
        embedder, reranker = self.get_models(self.model_args)
        if isinstance(embedder, ReasonIREncoder):
            retriever = ReasonIRRetriever(
                embedder,
                search_top_k=self.eval_args.search_top_k,
                overwrite=self.eval_args.overwrite
            )
        else:
            retriever = EvalDenseRetriever(
                embedder,
                search_top_k=self.eval_args.search_top_k,
                overwrite=self.eval_args.overwrite
            )
        if reranker is not None:
            reranker = EvalReranker(reranker, rerank_top_k=self.eval_args.rerank_top_k)
        return retriever, reranker

    def run(self):
        """
        Run the whole evaluation.
        """
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)

        if len(dataset_names) == 0:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the default dataset.")
            self.evaluator(
                splits=self.eval_args.splits,
                search_results_save_dir=self.eval_args.output_dir,
                retriever=self.retriever,
                reranker=self.reranker,
                corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values
            )
            logger.info(f"{self.eval_args.eval_name} evaluation completed.")
        else:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the following dataset names: {dataset_names}")
            for dataset_name in dataset_names:
                if self.eval_args.use_special_instructions:
                    self.retriever.stop_multi_process_pool()
                    self.retriever.embedder.query_instruction_for_retrieval = R2MEDInstructions[dataset_name]
                logger.info(f"Running {self.eval_args.eval_name} evaluation on: {dataset_name}")
                self.evaluator(
                    splits=self.eval_args.splits,
                    search_results_save_dir=self.eval_args.output_dir,
                    retriever=self.retriever,
                    reranker=self.reranker,
                    corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                    ignore_identical_ids=self.eval_args.ignore_identical_ids,
                    k_values=self.eval_args.k_values,
                    dataset_name=dataset_name,
                )
            logger.info(f"{self.eval_args.eval_name} evaluation on {dataset_names} completed.")

        logger.info("Start computing metrics.")
        self.evaluate_metrics(
            search_results_save_dir=self.eval_args.output_dir,
            output_method=self.eval_args.eval_output_method,
            output_path=self.eval_args.eval_output_path,
            metrics=self.eval_args.eval_metrics
        )
