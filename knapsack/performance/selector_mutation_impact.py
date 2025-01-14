import matplotlib.pyplot as plt

from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.selector import Selector


def selector_diversity_impact(
    algorithm: GeneticAlgorithm,
    selectors: list[Selector],
    mutation_rates: list[float],
    iterations=10,
):
    """Test the impact of mutation on diversity grouped by selectors.

    Args:
        algorithm: Genetic algorithm instance.
        selectors (list): List of selector objects to test.
        mutation_rates (list): List of mutation rates to test.
        iterations (int): Number of test iterations for each combination.

    Returns:
        dict: Nested dictionary with selectors as keys and mutation rate results as sub-keys.
    """
    diversity_by_selector = _measure_diversity_by_selector(
        algorithm, selectors, mutation_rates, iterations
    )
    plot_selector_diversity(diversity_by_selector)


def _measure_diversity_by_selector(
    algorithm: GeneticAlgorithm,
    selectors: list[Selector],
    mutation_rates: list[float],
    iterations=10,
):
    diversity_results = {}
    for selector in selectors:
        algorithm.selector = selector
        selector_name = selector.__class__.__name__
        diversity_results[selector_name] = {}

        for mutation_rate in mutation_rates:
            algorithm.mutation_operator.probability = mutation_rate
            diversity_per_run = []

            for _ in range(iterations):
                algorithm.evolve()
                diversity = algorithm.population.measure_diversity()
                diversity_per_run.append(diversity)

            diversity_results[selector_name][mutation_rate] = diversity_per_run

    return diversity_results


def plot_selector_diversity(results):
    """Plot the impact of mutation rates on diversity grouped by selectors."""
    plt.figure(figsize=(15, 10))

    for selector, mutation_data in results.items():
        plt.subplot(2, 2, list(results.keys()).index(selector) + 1)
        for rate, diversities in mutation_data.items():
            plt.plot(
                range(1, len(diversities) + 1),
                diversities,
                label=f"Mutation Rate {rate:.2f}",
                marker="o",
            )
        plt.title(f"Selector: {selector}")
        plt.xlabel("Iteration")
        plt.ylabel("Diversity (%)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
