
import random
import warnings
from itertools import chain, zip_longest
import joblib

import numba
import numpy as np

from hopwise.data import Interaction
from hopwise.data.dataset import KnowledgePathDataset
from hopwise.data.utils import user_parallel_sampling
from hopwise.utils import PathLanguageModelingTokenType, progress_bar, set_color

class KGGLMDataset(KnowledgePathDataset):
    def _get_field_from_config(self):
        super()._get_field_from_config()
        self.train_stage = self.config["train_stage"]

        path_sample_args = self.config["path_sample_args"]
        self.pretrain_hop_length = path_sample_args["pretrain_hop_length"]
        self.pretrain_hop_length = tuple(map(int, self.pretrain_hop_length[1:-1].split(",")))
        self.pretrain_paths = path_sample_args["pretrain_paths"]

    def generate_user_path_dataset(self):
        if self.train_stage == "pretrain":
            self.generate_pretrain_dataset()
        else:
            super().generate_user_path_dataset()

    def generate_pretrain_dataset(self):
        """Generate pretrain dataset for KGGLM model."""
        import pickle
        # Check if cached paths exist
        cache_file = f"./paths/KGGLM-{self.dataset_name}-pretrain-paths.pkl2"
        try:
            with open(cache_file, "rb") as f:
                self._path_dataset = pickle.load(f)
                self.logger.info(f"Loaded cached pretrain paths from {cache_file}")
                return
        except FileNotFoundError:
            self.logger.info(f"No cached pretrain paths found at {cache_file}. Generating new paths...")
        if self._path_dataset is None:
            graph = self._create_ckg_igraph(show_relation=True, directed=False)
            kg_rel_num = len(self.relations)
            graph.es["weight"] = [0.0] * (self.inter_num) + [1.0] * kg_rel_num

            # Remove zero-weight interaction edges before random-walk sampling.
            # They are never traversed (weight=0) but inflate average vertex degree
            # ~10x, making every weighted random_walk step scan far more neighbors
            # than necessary.  Deleting them reduces degree from ~90 to ~9 for onion
            # and gives a proportional speedup without changing which paths are found.
            # We keep the full graph (with interaction edges) for _add_paths_relations
            # so relation lookups remain correct.
            kg_only_graph = graph.copy()
            kg_only_graph.delete_edges(kg_only_graph.es.select(weight_eq=0))

            graph_min_iid = self.user_num
            min_hop, max_hop = self.pretrain_hop_length

            paths = set()
            iter_paths = progress_bar(
                range(graph_min_iid, len(kg_only_graph.vs)),
                ncols=100,
                total=len(kg_only_graph.vs) - graph_min_iid,
                desc=set_color("KGGLM Pre-training Path Sampling", "red", progress=True),
            )
            max_tries_per_entity = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]

            kwargs = dict(
                graph=kg_only_graph,
                min_hop=min_hop,
                max_hop=max_hop,
                pretrain_paths=self.pretrain_paths,
                max_tries_per_entity=max_tries_per_entity,
                paths=paths,
            )

            if not self.parallel_max_workers:
                for entity in iter_paths:
                    _generate_paths_random_walks(entity, **kwargs)
            else:
                list(joblib.Parallel(n_jobs=self.parallel_max_workers, prefer="threads")(
                    joblib.delayed(_generate_paths_random_walks)(entity, **kwargs) for entity in iter_paths
                ))

            paths_with_relations = self._add_paths_relations(graph, paths)

            lines = [self._format_path(path) for path in paths_with_relations]
            self._path_dataset = "\n".join(lines)
            
            # Cache the formatted strings for next time
            cache_file = f"./paths/KGGLM-{self.dataset_name}-pretrain-paths.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self._path_dataset, f)


def _generate_paths_random_walks(start_node, **kwargs):
    graph = kwargs.get("graph", )
    min_hop = kwargs.get("min_hop", 5)
    max_hop = kwargs.get("max_hop", 5)
    pretrain_paths = kwargs.get("pretrain_paths", 5)
    max_tries_per_entity = kwargs.get("max_tries_per_entity", 10)
    paths = kwargs.get("paths")
    
    # Use local set for faster duplicate checking per entity
    local_paths = set()
    
    for _ in range(pretrain_paths):
        path_hop_length = np.random.randint(min_hop, max_hop + 1)
        tries_per_entity = max_tries_per_entity

        while tries_per_entity > 0:
            generated_path = graph.random_walk(start_node, path_hop_length)
            generated_path = tuple(generated_path)
            # Check local set first (much faster), then global
            if generated_path not in local_paths:
                local_paths.add(generated_path)
                break
            tries_per_entity -= 1
    
            # Add all local paths to global set at once (thread-safe with GIL)
    paths.update(local_paths)