# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/18
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""hopwise.evaluator.collector
################################################
"""

import copy

import torch

from hopwise.utils import PathLanguageModelingTokenType

from hopwise.evaluator.register import Register, Register_KG
from hopwise.evaluator.utils import train_tsne


class DataStruct:
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value):
        if name not in self._data_dict:
            if isinstance(value, torch.Tensor):
                self._data_dict[name] = value.clone().detach()
            else:
                self._data_dict[name] = value
        elif isinstance(self._data_dict[name], torch.Tensor):
            self._data_dict[name] = torch.cat((self._data_dict[name], value.clone().detach()), dim=0)
        else:
            self._data_dict[name] = self._data_dict[name] + value

    def __str__(self):
        data_info = "\nContaining:\n"
        for data_key in self._data_dict.keys():
            data_info += data_key + "\n"
        return data_info


class Collector:
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = "full" in config["eval_args"]["mode"]
        self.topk = self.config["topk"]
        self.device = self.config["device"]

    def train_data_collect(self, train_data):
        """Collect the evaluation resource from training data.

        Args:
            train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need("data.num_items"):
            item_id = self.config["ITEM_ID_FIELD"]
            self.data_struct.set("data.num_items", train_data.dataset.num(item_id))
        if self.register.need("data.num_users"):
            user_id = self.config["USER_ID_FIELD"]
            self.data_struct.set("data.num_users", train_data.dataset.num(user_id))
        if self.register.need("data.count_items"):
            self.data_struct.set("data.count_items", train_data.dataset.item_counter)
        if self.register.need("data.count_users"):
            self.data_struct.set("data.count_users", train_data.dataset.user_counter)
        if self.register.need("data.history_index"):
            row = train_data.dataset.inter_feat[train_data.dataset.uid_field]
            col = train_data.dataset.inter_feat[train_data.dataset.iid_field]
            self.data_struct.set("data.history_index", torch.vstack([row, col]))
        if self.register.need("data.timestamp"):
            temporal_matrix = train_data.dataset.inter_matrix(value_field=train_data.dataset.time_field).toarray()
            self.data_struct.set("data.timestamp", temporal_matrix)

    def eval_data_collect(self, eval_data):
        """Collect the evaluation resource from evaluation data, such as user and item features.

        Args:
            eval_data (AbstractDataLoader): the evaluation dataloader which contains the evaluation data.

        """
        if self.register.need("eval_data.user_feat"):
            if not hasattr(eval_data.dataset, "user_feat") or eval_data.dataset.user_feat is None:
                raise AttributeError("Evaluation data does not include user features.")
            self.data_struct.set("eval_data.user_feat", eval_data.dataset.user_feat)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=self.device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=self.device).repeat(width).reshape(width, -1).transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = 0.5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self,
        scores,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.

        Args:
            scores (Torch.Tensor): the output tensor of model with the shape of `(N, )`
            interaction (Interaction): batched eval data.
            positive_u (Torch.Tensor): the row index of positive items for each user.
            positive_i (Torch.Tensor): the positive item id for each user.
        """
        if self.register.need("rec.users"):
            uid_field = self.config["USER_ID_FIELD"]
            self.data_struct.update_tensor("rec.users", interaction[uid_field])

        if self.register.need("rec.items"):
            # get topk
            _, topk_idx = torch.topk(scores, max(self.topk), dim=-1)  # n_users x k
            self.data_struct.update_tensor("rec.items", topk_idx)

        if self.register.need("rec.topk"):
            _, topk_idx = torch.topk(scores, max(self.topk), dim=-1)  # n_users x k
            pos_matrix = torch.zeros_like(scores, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.topk", result)

        if self.register.need("rec.meanrank"):
            desc_scores, desc_index = torch.sort(scores, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.meanrank", result)

        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", scores)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("data.label", interaction[self.label_field].to(self.device))

    def model_collect(self, model: torch.nn.Module, load_best_model=False):
        """Collect the evaluation resource from model and do something with the model.

        Args:
            model (nn.Module): the trained recommendation model.
            load_best_model (bool): whether to load the best model.
        """

        if self.config["tsne"] is not None:
            train_tsne(model, self.config["tsne"], load_best_model)

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """Collect the evaluation resource from total output and label.
        It was designed for those models that can not predict with batch.

        Args:
            eval_pred (torch.Tensor): the output score tensor of model.
            data_label (torch.Tensor): the label tensor.
        """
        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", eval_pred)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("data.label", data_label.to(self.device))

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        for key in self.data_struct._data_dict:
            if isinstance(self.data_struct._data_dict[key], torch.Tensor):
                self.data_struct._data_dict[key] = self.data_struct._data_dict[key].cpu()

        returned_struct = copy.deepcopy(self.data_struct)
        for key in [
            "rec.topk",
            "rec.semantic_topk",
            "rec.meanrank",
            "rec.score",
            "rec.items",
            "data.label",
            "rec.paths",
        ]:
            if key in self.data_struct:
                del self.data_struct[key]

        returned_struct.set("topk", self.topk)

        return returned_struct


class Collector_KG(Collector):
    """This collector is used to collect the resource for evaluator in knowledge graph embedding models.
    Specifically, it collects the predictions for the link prediction task, extending Collector from recommendation.

    """

    def __init__(self, config):
        super().__init__(config)
        self.register = Register_KG(config)
        self.topk = self.config["topk_kg"]


class ExplainableCollector(Collector):
    """This collector is used to collect the resource for evaluator in explainable recommendation models.
    It collects the KG paths and explanations for the recommendations made by the model and enables
    path quality evaluation.

    """

    def __init__(self, config):
        super().__init__(config)
        self.register = Register(config)

    def train_data_collect(self, train_data):
        super().train_data_collect(train_data)

        if self.register.need("data.max_path_type"):
            self.data_struct.set("data.max_path_type", torch.arange(train_data.dataset.relation_num))
        if self.register.need("data.node_degree"):
            self.data_struct.set("data.node_degree", self.node_degree_dict(train_data))
        if self.register.need("data.max_path_length"):
            if hasattr(train_data.dataset, "token_sequence_length"):
                # PEARLM or KGGLM
                sampled_path_len = train_data.dataset.token_sequence_length - 2
                self.data_struct.set("data.max_path_length", sampled_path_len)
            else:
                # PGPR or CAFE
                self.data_struct.set(
                    "data.max_path_length", max([len(path) for path in self.config["path_constraint"]]) * 2 - 1
                )

        if self.register.need("data.rid2relation"):
            self.data_struct.set("data.rid2relation", train_data.dataset.field2id_token["relation_id"])

    def node_degree_dict(self, train_data):
        # from pgpr knowledge graph
        # https://github.com/giacoballoccu/rep-path-reasoning-recsys/blob/main/models/PGPR/knowledge_graph.py
        aug_kg = train_data.dataset.ckg_dict_graph()
        degrees = {}
        for etype in aug_kg:
            degrees[etype] = {}
            for eid in aug_kg[etype]:
                count = 0
                for r in aug_kg[etype][eid]:
                    count += len(aug_kg[etype][eid][r])
                degrees[etype][eid] = count
        return degrees

    def eval_batch_collect(
        self,
        explanations,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.

        Args:
            explanations (tuple): a tuple containing the scores and paths, where:
                - scores (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                - paths (list): a list of quadruples representing the paths for each user.
            interaction (Interaction): batched eval data.
            positive_u (Torch.Tensor): the row index of positive items for each user.
            positive_i (Torch.Tensor): the positive item id for each user.
        """
        scores, paths = explanations
        super().eval_batch_collect(scores, interaction, positive_u, positive_i)

        if self.register.need("rec.paths"):
            self.data_struct.update_tensor("rec.paths", paths)


class SPRIGSemanticCollector(ExplainableCollector):
    """Collector that computes semantic-ID top-k matches for SPRIG."""

    def __init__(self, config):
        super().__init__(config)
        self.semantic_vocab = None
        self.tokenizer = None
        self.semantic_ids_per_item = config["semantic_ids_per_item"]
        self.sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token

        if any(metric.lower() in {"hit", "mrr", "ndcg"} for metric in config["metrics"]):
            setattr(self.register, "rec.semantic_topk", True)

    def set_semantic_resources(self, dataset):
        if hasattr(dataset, "semantic_vocab"):
            self.semantic_vocab = dataset.semantic_vocab
        if hasattr(dataset, "tokenizer"):
            self.tokenizer = dataset.tokenizer
        if self.semantic_ids_per_item is None and self.semantic_vocab is not None:
            self.semantic_ids_per_item = self.semantic_vocab.num_semantic_tokens_per_item()

    def train_data_collect(self, train_data):
        super().train_data_collect(train_data)
        self.set_semantic_resources(train_data.dataset)

    def _extract_last_sem_block(self, seq_tokens):
        if self.tokenizer is None:
            return None
        if self.semantic_ids_per_item is None:
            return None

        last_block = None
        i = 0
        while i < len(seq_tokens):
            if seq_tokens[i].startswith(self.sem_prefix):
                block = []
                j = i
                while j < len(seq_tokens) and seq_tokens[j].startswith(self.sem_prefix) and len(block) < self.semantic_ids_per_item:
                    block.append(seq_tokens[j])
                    j += 1
                if len(block) == self.semantic_ids_per_item:
                    last_block = block
                i = j
            else:
                i += 1

        if last_block is None:
            return None

        token_ids = tuple(self.tokenizer.convert_tokens_to_ids(tok) for tok in last_block)
        if hasattr(self.tokenizer, "unk_token_id") and self.tokenizer.unk_token_id in token_ids:
            return None
        return token_ids

    def eval_batch_collect(
        self,
        explanations,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        scores, paths = explanations
        super().eval_batch_collect(explanations, interaction, positive_u, positive_i)

        if not self.register.need("rec.semantic_topk"):
            return

        if self.semantic_vocab is None or self.tokenizer is None:
            raise ValueError("SPRIG semantic evaluation requires semantic_vocab and tokenizer.")

        uid_field = self.config["USER_ID_FIELD"]
        batch_user_ids = interaction[uid_field].tolist()
        uid_to_batch = {int(uid): idx for idx, uid in enumerate(batch_user_ids)}

        user_candidates = {idx: [] for idx in range(scores.size(0))}
        for uid, _, seq_score, seq_tokens in paths:
            batch_idx = uid_to_batch.get(int(uid))
            if batch_idx is None:
                continue
            sem_tuple = self._extract_last_sem_block(seq_tokens)
            if sem_tuple is None:
                continue
            user_candidates[batch_idx].append((float(seq_score), sem_tuple))

        user_pos_sem = {idx: set() for idx in range(scores.size(0))}
        for u, i in zip(positive_u.tolist(), positive_i.tolist()):
            if i not in self.semantic_vocab:
                continue
            sem_tuple = tuple(self.semantic_vocab.get_item_tokens(i))
            user_pos_sem[int(u)].add(sem_tuple)

        max_k = max(self.topk)
        pos_idx = torch.zeros((scores.size(0), max_k), dtype=torch.int, device=self.device)
        pos_len_list = torch.zeros((scores.size(0), 1), dtype=torch.int, device=self.device)

        for batch_idx in range(scores.size(0)):
            pos_set = user_pos_sem.get(batch_idx, set())
            pos_len_list[batch_idx, 0] = len(pos_set)
            if not pos_set:
                continue

            candidates = sorted(user_candidates.get(batch_idx, []), key=lambda x: x[0], reverse=True)
            seen = set()
            ranked = []
            for _, sem_tuple in candidates:
                if sem_tuple in seen:
                    continue
                seen.add(sem_tuple)
                ranked.append(sem_tuple)
                if len(ranked) >= max_k:
                    break

            for j, sem_tuple in enumerate(ranked):
                if sem_tuple in pos_set:
                    pos_idx[batch_idx, j] = 1

        result = torch.cat((pos_idx, pos_len_list), dim=1)
        self.data_struct.update_tensor("rec.semantic_topk", result)
