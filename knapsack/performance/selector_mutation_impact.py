from knapsack.analyze.utility import ExperimentConfig, ExperimentResults, init_alg
import matplotlib.pyplot as plt

from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.selector import Selector


def selector_diversity_impact(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    algorithm = init_alg(alg, config)
    diversity_by_selector = _measure_diversity_by_selector(algorithm, config, iteratons)
    plot_selector_diversity(diversity_by_selector)


def _measure_diversity_by_selector(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
    iterations=10,
):
    diversity_results = {}
    for selector in config.selectors:
        alg.selector = selector

        key = selector.__class__.__name__
        diversity_results[key] = {}

        for mutation_rate in config.mutation_rates:
            alg.mutation_rate = mutation_rate

            diversity_per_run = []
            for _ in range(iterations):
                alg.evolve()
                diversity = alg.population.measure_diversity()
                diversity_per_run.append(diversity)
                alg.reinitialize_population()

            diversity_results[key][mutation_rate] = diversity_per_run

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
