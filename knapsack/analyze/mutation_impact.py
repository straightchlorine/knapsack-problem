import matplotlib.pyplot as plt

from knapsack.analyze.utility import ExperimentConfig, init_alg
from knapsack.genetic_algorithm import GeneticAlgorithm


def mutation_impact(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    algorithm = init_alg(alg, config)
    diversity = measure_mutation_impact(algorithm, config, iterations)
    plot_population_diversity(diversity)
    return diversity


def measure_mutation_impact(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
    iterations=10,
):
    diversity_results = {}
    for mutation_rate in config.mutation_rates:
        alg.mutation_operator.probability = mutation_rate
        alg.clear_metrics()
        diversity_per_run = []

        for _ in range(iterations):
            alg.evolve()
            diversity = alg.population.measure_diversity()
            diversity_per_run.append(diversity)
        diversity_results[mutation_rate] = diversity_per_run

    return diversity_results


def plot_population_diversity(results):
    """Plot the impact of mutation rates on diversity."""
    plt.figure(figsize=(10, 6))
    for rate, diversities in results.items():
        plt.plot(
            range(1, len(diversities) + 1),
            diversities,
            label=f"Mutation Rate {rate:.2f}",
            marker="o",
        )

    plt.title("Impact of Mutation on Genetic Diversity")
    plt.xlabel("Iteration")
    plt.ylabel("Diversity (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
