
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
        import os, pickle

        if self._path_dataset is None:
            graph = self._create_ckg_igraph(show_relation=True, directed=False)
            kg_rel_num = len(self.relations)
            graph.es["weight"] = [0.0] * (self.inter_num) + [1.0] * kg_rel_num
            walk_graph = graph.subgraph_edges(
                graph.es.select(weight_eq=1.0), delete_vertices=False
            )

            graph_min_iid = 1 + self.user_num
            min_hop, max_hop = self.pretrain_hop_length

            paths = set()
            iter_paths = progress_bar(
                range(graph_min_iid + 1, len(graph.vs)),
                ncols=100,
                total=len(graph.vs) - graph_min_iid,
                desc=set_color("KGGLM Pre-training Path Sampling", "red", progress=True),
            )
            max_tries_per_entity = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]

            kwargs = dict(
                graph=walk_graph,
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
                # Fixed: removed return_as="generator" and consume results properly
                list(joblib.Parallel(n_jobs=self.parallel_max_workers, prefer="threads")(
                    joblib.delayed(_generate_paths_random_walks)(entity, **kwargs) for entity in iter_paths
                ))

            paths_with_relations = self._add_paths_relations(graph, paths)

            path_string = ""
            for path in paths_with_relations:
                path_string += self._format_path(path) + "\n"
                
            self._path_dataset = path_string


def _generate_paths_random_walks(start_node, **kwargs):
    graph = kwargs.get("graph")
    min_hop = kwargs.get("min_hop")
    max_hop = kwargs.get("max_hop")
    pretrain_paths = kwargs.get("pretrain_paths")
    max_tries_per_entity = kwargs.get("max_tries_per_entity")
    paths = kwargs.get("paths")

    # Local rebinds: cheaper attribute/global lookups in the hot loop.
    rw = graph.random_walk

    # Vectorize hop-length sampling instead of one randint() per path.
    hop_lengths = np.random.randint(min_hop, max_hop + 1, size=pretrain_paths)

    local_paths = set()
    local_add = local_paths.add
    local_contains = local_paths.__contains__

    for path_hop_length in hop_lengths:
        hop = int(path_hop_length)
        for _ in range(max_tries_per_entity):
            # No `weights=` -> igraph picks a uniform neighbor in C, fast path.
            generated_path = tuple(rw(start_node, hop))
            if not local_contains(generated_path):
                local_add(generated_path)
                break

    # set.update is atomic under the GIL, so this is safe with prefer="threads".
    paths.update(local_paths)