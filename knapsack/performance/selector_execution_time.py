import time
import matplotlib.pyplot as plt
import numpy as np

from knapsack.analyze.utility import (
    ExperimentConfig,
    init_alg,
    is_class_equal,
)
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.selector import Selector


def selector_time_efficiency(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    algorithm = init_alg(alg, config)
    exec_time = _measure(algorithm, config, iterations)
    plot_selector_execution_speed(exec_time)


def _measure(alg: GeneticAlgorithm, config: ExperimentConfig, iterations=10):
    speed_results = {}
    for selector in config.selectors:
        if not is_class_equal(alg.selector, selector):
            alg.selector = selector

        execution_times = []
        for _ in range(iterations):
            start_time = time.time()
            alg.evolve()
            execution_times.append(time.time() - start_time)
            alg.reinitialize_population()

        speed_results[selector.__class__.__name__] = execution_times
    return speed_results


def plot_selector_execution_speed(result):
    """Plot the execution speed for different selectors."""
    mean_times = {selector: np.mean(times) for selector, times in result.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(list(mean_times.keys()), list(mean_times.values()), color="skyblue")

    plt.title("Execution Speed for Selection Methods")
    plt.ylabel("Mean Execution Time (s)")
    plt.xlabel("Selection Method")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()
