import matplotlib.pyplot as plt
import numpy as np

from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.selector import Selector


def selector_effectiveness(
    algorithm: GeneticAlgorithm, selectors: list[Selector], iterations=10
):
    """Test how effective each selector is at procuring high fitness solutions.

    Args:
        algorithm: Genetic algorithm instance.
        selectors (list): List of selector objects to test.
        iterations (int): Number of test iterations for each selector.

    Returns:
        dict: Dictionary containing fitness results for each selector.
    """
    results = _selection_performance_analysis(algorithm, selectors, iterations)
    plot_selector_performance(results)
    selection_performance_metrics(results)
    return results


def _selection_performance_analysis(
    algorithm: GeneticAlgorithm, selectors: list[Selector], iterations=10
):
    """Test different selection methods and collect data about quality of the solutions."""
    results = {}
    for selector in selectors:
        algorithm.selector = selector
        solutions = []
        for _ in range(iterations):
            algorithm.evolve()
            best_solution = algorithm.get_best_solution()
            solutions.append(algorithm.get_solution_fitness(best_solution))
        results[selector.__class__.__name__] = solutions
    return results


def plot_selector_performance(results):
    """Plot the comparison of selection methods."""
    plt.figure(figsize=(10, 6))

    for method, fitness_values in results.items():
        plt.plot(
            range(1, len(fitness_values) + 1),
            fitness_values,
            label=method,
            marker="o",
        )

    plt.title("Comparison of Selection Methods")
    plt.xlabel("Iteration")
    plt.ylabel("Best Solution Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


def selection_performance_metrics(results):
    """Print statistics for each selection method.

    Args:
        results (dict): Dictionary containing fitness results for each selector.
    """
    print("Selection Method Analysis:")
    print("=" * 40)
    for method, fitness_values in results.items():
        mean_fitness = np.mean(fitness_values)
        std_dev = np.std(fitness_values)
        print(
            f"{method}: Mean Fitness = {mean_fitness:.2f}, "
            f"Std Dev = {std_dev:.2f}"
        )
    print("=" * 40)
