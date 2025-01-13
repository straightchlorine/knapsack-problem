import matplotlib.pyplot as plt


def test_mutation_impact(algorithm, mutation_rates, iterations=10):
    """Test the impact of mutation on diversity.

    Args:
        algorithm: Genetic algorithm instance.
        mutation_rates (list): List of mutation rates to test.
        iterations (int): Number of test iterations for each mutation rate.

    Returns:
        dict: Dictionary containing diversity results for each mutation rate.
    """
    diversity_results = {}
    for mutation_rate in mutation_rates:
        algorithm.mutation_rate = mutation_rate
        diversity_per_run = []
        for _ in range(iterations):
            algorithm.evolve()
            diversity = algorithm.measure_diversity()
            diversity_per_run.append(diversity)
        diversity_results[mutation_rate] = diversity_per_run
    return diversity_results


def plot_mutation_diversity(diversity_results):
    """Plot the impact of mutation rates on diversity.

    Args:
        diversity_results (dict): Dictionary with mutation rates and diversity data.
    """
    plt.figure(figsize=(10, 6))
    for rate, diversities in diversity_results.items():
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
