import torch
from transformers.trainer import is_torch_xla_available, SaveStrategy

from typing import Dict
from FlagEmbedding.finetune.embedder.decoder_only.base import DecoderOnlyEmbedderTrainer

from modeling import DynQRIBiDecoderOnlyEmbedderModel


class DynQRIDecoderOnlyEmbedderTrainer(DecoderOnlyEmbedderTrainer):
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model: DynQRIBiDecoderOnlyEmbedderModel, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # log qri score stats
            cur_qri_score_stat = model.get_cur_qri_score_stat()
            
            if cur_qri_score_stat is not None:
                original_loss = cur_qri_score_stat.pop("original_loss")
                original_loss_scalar = self._nested_gather(original_loss).mean().item()
                logs["original_loss"] = round(original_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                
                logs.update(cur_qri_score_stat)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
