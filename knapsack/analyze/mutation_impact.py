import matplotlib.pyplot as plt

from knapsack.genetic_algorithm import GeneticAlgorithm


def mutation_impact(
    algorithm: GeneticAlgorithm, mutation_rates: list[float], iterations=10
):
    """Analyse the impact of mutation on diversity.

    Args:
        algorithm: Genetic algorithm instance.
        mutation_rates (list): List of mutation rates to test.
        iterations (int): Number of test iterations for each mutation rate.

    Returns:
        dict: Dictionary containing diversity results for each mutation rate.
    """
    diversity = measure_mutation_impact(algorithm, mutation_rates, iterations)
    plot_population_diversity(diversity)
    return diversity


def measure_mutation_impact(
    algorithm: GeneticAlgorithm, mutation_rates: list[float], iterations=10
):
    diversity_results = {}
    for mutation_rate in mutation_rates:
        algorithm.mutation_rate = mutation_rate
        diversity_per_run = []
        for _ in range(iterations):
            algorithm.evolve()
            diversity = algorithm.population.measure_diversity()
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
