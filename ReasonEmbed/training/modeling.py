import torch
from torch import Tensor
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
from transformers import PreTrainedModel, PreTrainedTokenizer
from FlagEmbedding.finetune.embedder.decoder_only.base.modeling import BiDecoderOnlyEmbedderModel

from dataclasses import dataclass
from typing import Dict, Optional, List, Union


@dataclass
class EmbedderOutput(ModelOutput):
    """
    Output information returned by the model.
    """
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class DynQRIBiDecoderOnlyEmbedderModel(BiDecoderOnlyEmbedderModel):
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'last_token',
        normalize_embeddings: bool = False,
        qri_start_step: int = -1,
        qri_score_mapping: str = "clamp",
        qri_score_clamp_min: float = 0.0,
        qri_score_clamp_max: float = 5.0,
        weights_norm: str = "l1",
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
            sentence_pooling_method=sentence_pooling_method,
            normalize_embeddings=normalize_embeddings,
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        
        self.qri_start_step = qri_start_step
        self.qri_score_mapping = qri_score_mapping
        self.qri_score_clamp_min = qri_score_clamp_min
        self.qri_score_clamp_max = qri_score_clamp_max
        self.weights_norm = weights_norm
        
        self.step = 0
        self.cur_qri_score_stat_dict = {
            "original_loss": None,
            "qri_score_min": None,
            "qri_score_max": None,
            "qri_score_mean": None,
            "qri_score_median": None,
        }
    
    def get_cur_qri_score_stat(self):
        if self.step > self.qri_start_step:
            return self.cur_qri_score_stat_dict
        else:
            return None
    
    def _update_cur_qri_score_stat(self, qri_scores, original_loss):
        qri_scores = qri_scores.cpu()
        self.cur_qri_score_stat_dict["original_loss"] = original_loss.detach().mean()
        self.cur_qri_score_stat_dict["qri_score_min"] = round(float(torch.min(qri_scores)), 4)
        self.cur_qri_score_stat_dict["qri_score_max"] = round(float(torch.max(qri_scores)), 4)
        self.cur_qri_score_stat_dict["qri_score_mean"] = round(float(torch.mean(qri_scores)), 4)
        self.cur_qri_score_stat_dict["qri_score_median"] = round(float(torch.median(qri_scores)), 4)
    
    def _map_qri_scores_to_weights(self, qri_scores, device):
        if self.qri_score_mapping == "clamp":
            # clamp scores to [qri_score_clamp_min, qri_score_clamp_max]
            weights = torch.clamp(qri_scores, min=self.qri_score_clamp_min, max=self.qri_score_clamp_max)
        elif self.qri_score_mapping == "none":
            weights = qri_scores
        else:
            raise ValueError(f"Invalid qri_score_mapping: {self.qri_score_mapping}")
        
        weights = weights.to(device)
        
        if self.weights_norm == "l1":
            # normalize weights to sum = 1.0
            weights = weights / torch.sum(weights)
        elif self.weights_norm == "none":
            pass
        else:
            raise ValueError(f"Invalid weights_norm: {self.weights_norm}")
        
        return weights
    
    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        reasoning_queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        """The computation performed at every call.

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input queries. Defaults to ``None``.
            reasoning_queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input reasoning queries. Defaults to ``None``.
            passages (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input passages. Defaults to ``None``.
            teacher_scores (Union[None, List[float]], optional): Teacher scores for distillation. Defaults to ``None``.
            no_in_batch_neg_flag (bool, optional): If True, use no in-batch negatives and no cross-device negatives. Defaults to ``False``.

        Returns:
            EmbedderOutput: Output of the forward call of model.
        """
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * group_size, dim)
        
        if self.step >= self.qri_start_step:
            with torch.no_grad():
                reasoning_q_reps = self.encode(reasoning_queries) # (batch_size, dim)
        else:
            reasoning_q_reps = None

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()   # (batch_size, group_size)
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # (batch_size, group_size)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
            
            if self.step >= self.qri_start_step:
                with torch.no_grad():
                    reasoning_scores, reasoning_loss = compute_loss_func(reasoning_q_reps, p_reps.detach(), teacher_targets=teacher_targets)
                    
                    # compute qri scores
                    qri_scores = loss.detach() / (reasoning_loss + 1e-6)
                    self._update_cur_qri_score_stat(qri_scores, loss.detach())
                    
                    # map qri scores to weights
                    weights = self._map_qri_scores_to_weights(qri_scores, device=loss.device)
            
                # apply weights to loss. NOTE: weights already normalized
                final_loss = (loss * weights).sum()
            else:
                final_loss = loss.mean()
            
            self.step += 1
        else:
            final_loss = None

        return EmbedderOutput(
            loss=final_loss,
        )
