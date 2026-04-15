from hopwise.evaluator import SPRIGEvaluator
from hopwise.evaluator.collector import SPRIGSemanticCollector
from hopwise.trainer.trainer import KGGLMTrainer


class SPRIGTrainer(KGGLMTrainer):
    r"""SPRIGTrainer is designed for SPRIG, which extends KGGLM's path-based
    language modelling with semantic IDs.

    The training workflow (pretrain -> finetune, HF Trainer integration,
    checkpoint management) is identical to KGGLM.  This subclass exists so
    that ``get_trainer(MODEL_TYPE, "SPRIG")`` resolves to a dedicated class,
    keeping the door open for SPRIG-specific overrides in the future.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.eval_collector = SPRIGSemanticCollector(config)
        self.evaluator = SPRIGEvaluator(config)

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if eval_data:
            self.eval_collector.set_semantic_resources(eval_data._dataset)
        return super().evaluate(
            eval_data,
            load_best_model=load_best_model,
            model_file=model_file,
            show_progress=show_progress,
        )
