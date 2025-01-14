import time
import matplotlib.pyplot as plt
import numpy as np

from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.selector import Selector


def selector_time_efficiency(
    alg: GeneticAlgorithm, selectors: list[Selector], iterations=10
):
    """Measure the execution time for each selector.

    Args:
        algorithm: Genetic algorithm instance.
        selectors (list[Selector]): List of selector objects to test.
        iterations (int): Number of test iterations for each selector.

    Returns:
        dict: Dictionary containing execution times for each selector.
    """
    exec_time = _measure(alg, selectors, iterations)
    plot_selector_execution_speed(exec_time)


def _measure(
    algorithm: GeneticAlgorithm, selectors: list[Selector], iterations=10
):
    speed_results = {}
    for selector in selectors:
        algorithm.selector = selector
        execution_times = []
        for _ in range(iterations):
            start_time = time.time()
            algorithm.evolve()
            execution_times.append(time.time() - start_time)
        speed_results[selector.__class__.__name__] = execution_times
    return speed_results


def plot_selector_execution_speed(result):
    """Plot the execution speed for different selectors."""
    mean_times = {
        selector: np.mean(times) for selector, times in result.items()
    }

    plt.figure(figsize=(10, 6))
    plt.bar(
        list(mean_times.keys()), list(mean_times.values()), color="skyblue"
    )

    plt.title("Execution Speed for Selection Methods")
    plt.ylabel("Mean Execution Time (s)")
    plt.xlabel("Selection Method")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()
