import sys
import time

import deglib
import numpy as np

from ..base.module import BaseANN


class ProgressCallback:
    def __init__(
            self, num_new_entries: int, num_remove_entries: int, bar_length: int = 60, min_print_interval: float = 0.1
    ):
        self.num_new_entries = num_new_entries
        self.num_remove_entries = num_remove_entries
        self.bar_length = bar_length
        self.maximal = self.num_new_entries + self.num_remove_entries
        self.len_max = len(str(self.maximal))
        self.last_print_time = 0
        self.min_print_interval = min_print_interval

    def __call__(self, builder_status):
        current_time = time.time()
        num_steps = builder_status.added + builder_status.deleted
        last_step = num_steps == self.maximal
        if current_time - self.last_print_time >= self.min_print_interval or last_step:
            self.last_print_time = current_time

            progress = num_steps / self.maximal
            block = int(self.bar_length * progress)
            bar = '#' * block + '-' * (self.bar_length - block)
            percentage = progress * 100
            sys.stdout.write(f'\n{percentage:6.2f}% [{bar}] ({num_steps:{self.len_max}} / {self.maximal})')
            if last_step:
                sys.stdout.write('\n')  # newline at the end
            sys.stdout.flush()


def build_from_data(
        data, edges_per_vertex = 32, 
        metric = deglib.Metric.L2, optimization_target = deglib.builder.OptimizationTarget.LowLID,
        extend_k = None, extend_eps = 0.2,
        improve_k = 0, improve_eps = 0.001, max_path_length = 10,
        swap_tries = 0, additional_swap_tries = 0, remove_edges = False
):
    print('edges_per_vertex', edges_per_vertex)
    print('metric', metric)
    print('optimization_target', optimization_target)
    print('extend_k', extend_k)
    print('extend_eps', extend_eps)
    print('improve_k', improve_k)
    print('improve_eps', improve_eps)
    print('max_path_length', max_path_length)
    print('swap_tries', swap_tries)
    print('additional_swap_tries', additional_swap_tries)

    graph = deglib.graph.SizeBoundedGraph.create_empty(data.shape[0], data.shape[1], edges_per_vertex, metric)
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, optimization_target=optimization_target, extend_k=extend_k, extend_eps=extend_eps, improve_k=improve_k,
        improve_eps=improve_eps, max_path_length=max_path_length, swap_tries=swap_tries,
        additional_swap_tries=additional_swap_tries
    )
    builder.set_thread_count(1)
    labels = np.arange(data.shape[0], dtype=np.uint32)
    builder.add_entry(labels, data)
    builder.build(callback=ProgressCallback(builder.get_num_new_entries(), builder.get_num_remove_entries()))

    if remove_edges:
        graph.remove_non_mrng_edges()

    return graph


class DegLib(BaseANN):
    def __init__(self, metric, method_params):
        self.metric = {"angular": deglib.Metric.InnerProduct, "euclidean": deglib.Metric.L2}[metric]
        self.method_params = method_params
        self.query_params = {}

    def fit(self, X):
        # fix optimization target datatype
        build_params = {}
        for key, value in self.method_params.items():
            if key == 'optimization_target':
                value = deglib.builder.OptimizationTarget(value)
            build_params[key] = value

        self.index = build_from_data(X, **build_params, metric=self.metric)

    def set_query_arguments(self, params):
        self.query_params = params
        self.name = "deglib (buildParams: {}, queryParams: {})".format(self.method_params, self.query_params)

    def query(self, v, n):
        v = np.expand_dims(v, axis=0)
        return self.index.search(v, k=n, **self.query_params)[0][0]

    def freeIndex(self):
        del self.index
