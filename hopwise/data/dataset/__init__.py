from hopwise.data.dataset.dataset import Dataset
from hopwise.data.dataset.sequential_dataset import SequentialDataset
from hopwise.data.dataset.kg_dataset import KnowledgeBasedDataset
from hopwise.data.dataset.kg_seq_dataset import KGSeqDataset
from hopwise.data.dataset.decisiontree_dataset import DecisionTreeDataset
from hopwise.data.dataset.kg_path_dataset import KnowledgePathDataset
from hopwise.data.dataset.customized_dataset import *
from hopwise.data.dataset.sprig_dataset import SPRIGDataset
from hopwise.data.dataset.sprigl_dataset import SPRIGLDataset
from hopwise.data.dataset.sprigt_dataset import SPRIGTDataset

# Boosted variants share the same dataset classes as their base models.
BoostedSPRIGDataset = SPRIGDataset
BoostedSPRIGLDataset = SPRIGLDataset

# SPRIGLC (SPRIGL + coverage-penalty logits processor) is decode-time only.
SPRIGLCDataset = SPRIGLDataset

# RoPE-backbone variants (SPRIGR/SPRIGLR) share the same dataset classes as
# their GPT2-backbone counterparts - the backbone swap is model-side only.
SPRIGRDataset = SPRIGDataset
SPRIGLRDataset = SPRIGLDataset
from hopwise.data.dataset.kgglm_dataset import KGGLMDataset
