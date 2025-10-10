from FlagEmbedding.abc.evaluation import EvalDenseRetriever


class ReasonIRRetriever(EvalDenseRetriever):
    def __str__(self):
        return str(self.embedder)
    
    def stop_multi_process_pool(self):
        pass
