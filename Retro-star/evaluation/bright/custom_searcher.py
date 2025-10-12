import os
import logging

from typing import Any, Dict, Optional
from FlagEmbedding.abc.evaluation import EvalRetriever


logger = logging.getLogger(__name__)

class CustomEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
    ):
        self.model_name_or_path = model_name_or_path
        self.embedder_model_class = "custom"
    

class CustomEvalRetriever(EvalRetriever):
    def __init__(
        self,
        embedder: CustomEmbedder,
        search_top_k: int = 1000,
        overwrite: bool = False
    ):
        self.embedder = embedder
        self.search_top_k = search_top_k
        self.overwrite = overwrite     

    def __str__(self) -> str:
        return os.path.basename(self.embedder.model_name_or_path)

    def __call__(self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        pass
    