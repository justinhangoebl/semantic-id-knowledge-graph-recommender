# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/16, 2020/8/6, 2020/8/25, 2023/4/24
# @Author : Zhen Tian, Shanlei Mu, Yupeng Hou, Chenglong Ma
# @Email  : chenyuwuxinn@gmail.com, slmu@ruc.edu.cn, houyupeng@ruc.edu.cn, chenglong.m@outlook.com

"""hopwise.model.abstract_recommender
##################################
"""

from logging import getLogger

import numpy as np
import torch
from torch import nn

from hopwise.model.layers import FLEmbedding, FMEmbedding, FMFirstOrderLinear
from hopwise.model.logits_processor import LogitsProcessorList
from hopwise.utils import (
    FeatureSource,
    FeatureType,
    GenerationOutputs,
    InputType,
    KnowledgeEvaluationType,
    ModelType,
    PathLanguageModelingTokenType,
    get_logits_processor,
    get_sequence_postprocessor,
    set_color,
)


class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def __init__(self, _skip_nn_module_init=False):
        self.logger = getLogger()

        if not _skip_nn_module_init:
            super().__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""Full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def full_sort_predict_kg(self, interaction):
        r"""Full sort prediction KG function.
        Given heads, calculate the scores between heads and all candidate tails.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given heads and all candidate tails,
            shape: [n_batch_heads * n_candidate_tails]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color("\nTrainable parameters", "blue") + f": {params}"

class SequentialRecommender(AbstractRecommender):
    """This is a abstract sequential recommender. All the sequential model should implement This class."""

    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super().__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class KnowledgeRecommender(AbstractRecommender):
    """This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
    The base knowledge-based recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.KNOWLEDGE

    def __init__(self, config, dataset, _skip_nn_module_init=False):
        super().__init__(_skip_nn_module_init=_skip_nn_module_init)

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.ENTITY_ID = config["ENTITY_ID_FIELD"]
        self.RELATION_ID = config["RELATION_ID_FIELD"]
        self.HEAD_ENTITY_ID = config["HEAD_ENTITY_ID_FIELD"]
        self.TAIL_ENTITY_ID = config["TAIL_ENTITY_ID_FIELD"]
        self.NEG_TAIL_ENTITY_ID = config["NEG_PREFIX"] + self.TAIL_ENTITY_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)

        # load parameters info
        if not _skip_nn_module_init:
            self.device = config["device"]


class ExplainableRecommender:
    """This is a abstract explainable-based recommender. All the explainable-based model should implement this class.
    This class use templates to make the explanation more interpretable.

    """

    def explain(self, interaction):
        r"""
        Explain the prediction function.

        Given users, calculate the scores and paths between users and all candidate items,
        then return the templates filled with path data.

        Args:
            interaction (Interaction): The interaction batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
                with shape [n_batch_users * n_candidate_items].
            pandas.DataFrame: Explanation of the prediction, containing paths and corresponding templates,
                with shape [n_paths * [uid, pid, score, template1, template2, ..., #templates]].
        """
        raise NotImplementedError("explain is not implemented")

    def decode_path(self, path):
        r"""
        Decode the path into a string. Path decoding is specific to each model.

        Args:
            path (list): The path data.

        Returns:
            str: The decoded path string.
        """
        raise NotImplementedError("decode_path is not implemented")


class PathLanguageModelingRecommender(KnowledgeRecommender):
    """This is an abstract path-language-modeling recommender.
    All the path-language-modeling model should implement this class.
    The base path-language-modeling recommender class inherits the knowledge-aware recommender class to
    learn from knowledge graph paths defined by a chain of entity-relation triplets.
    """

    type = ModelType.PATH_LANGUAGE_MODELING
    input_type = InputType.PATHWISE

    def __init__(self, config, dataset, _skip_nn_module_init=True):
        super().__init__(config, dataset, _skip_nn_module_init=_skip_nn_module_init)

        self.n_tokens = len(dataset.tokenizer)
        self.token_sequence_length = dataset.token_sequence_length - 1  # EOS token is not included

        logits_processor = get_logits_processor(config["model"])(
            tokenized_ckg=dataset.get_tokenized_ckg(),
            tokenized_used_ids=dataset.get_tokenized_used_ids(),
            max_sequence_length=self.token_sequence_length,
            tokenizer=dataset.tokenizer,
            task=KnowledgeEvaluationType.REC,
        )
        self.logits_processor_list = LogitsProcessorList([logits_processor])

        self.sequence_postprocessor = get_sequence_postprocessor(config["sequence_postprocessor"])(
            dataset.tokenizer,
            dataset.get_user_used_ids(),
            dataset.item_num,
            topk=config["topk"],
        )

    @torch.no_grad()
    def generate(self, inputs, top_k=None, paths_per_user=1, **kwargs):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            inputs (dict): A dictionary containing the input_ids tensor with shape (b, t).
            top_k (int, optional): If specified, only the top k logits will be considered
                for sampling at each step. Defaults to None.
            paths_per_user (int, optional): How many paths to return for each user.
            **kwargs: Additional keyword arguments for the model. In future, it can be used to pass
                other generation parameters such as temperature, repetition penalty, etc.
        """
        max_new_tokens = self.token_sequence_length - inputs["input_ids"].size(1)

        # How many paths to return?
        inputs["input_ids"] = inputs["input_ids"].repeat_interleave(paths_per_user, dim=0)
        scores = torch.full((inputs["input_ids"].size(0), max_new_tokens, self.n_tokens), -torch.inf).to(self.device)
        for i in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.predict(inputs)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature

            # KGCD
            logits = self.logits_processor_list(inputs["input_ids"], logits)

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores[:, i] = probs
            # sample from the distribution
            path_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            inputs["input_ids"] = torch.cat((inputs["input_ids"], path_next), dim=1)

        return GenerationOutputs(sequences=inputs["input_ids"], scores=torch.unbind(scores, dim=1))


class ExplainablePathLanguageModelingRecommender(PathLanguageModelingRecommender, ExplainableRecommender):
    """This is an abstract explainable path-language-modeling recommender.
    All the explainable path-language-modeling model should implement this class.
    The base explainable path-language-modeling recommender class inherits the path-language-modeling recommender class
    to learn from knowledge graph paths defined by a chain of entity-relation triplets.
    """

    def __init__(self, config, dataset, _skip_nn_module_init=True):
        super().__init__(config, dataset, _skip_nn_module_init=_skip_nn_module_init)

    def explain(self, inputs, **kwargs):
        kwargs["max_length"] = self.token_sequence_length
        kwargs["min_length"] = self.token_sequence_length
        outputs = self.generate(inputs, **kwargs)

        max_new_tokens = self.token_sequence_length - inputs["input_ids"].size(1)

        scores, sequences = self.sequence_postprocessor.get_sequences(outputs, max_new_tokens=max_new_tokens)

        for seq in sequences:
            seq[-1] = self.decode_path(seq[-1])

        return scores, sequences

    def decode_path(self, path):
        """Standardize path format"""
        new_path = []
        # Process the path
        # [BOS] U R I R E/I R I
        for node_idx in range(1, len(path) + 1, 2):
            if path[node_idx].startswith(PathLanguageModelingTokenType.USER.token):
                user_id = int(path[node_idx][1:])
                if node_idx - 1 == 0:
                    relation = "self_loop"
                else:
                    relation = int(path[node_idx - 1][1:])

                new_node = (relation, "user", user_id)
            elif path[node_idx].startswith(PathLanguageModelingTokenType.ITEM.token):
                relation = int(path[node_idx - 1][1:])
                item_id = int(path[node_idx][1:])
                new_node = (relation, "item", item_id)
            else:
                # Is an entity
                relation = int(path[node_idx - 1][1:])
                entity_id = int(path[node_idx][1:])
                new_node = (relation, "entity", entity_id)
            new_path.append(new_node)
        return new_path
