import matplotlib.pyplot as plt

from knapsack.genetic_algorithm import GeneticAlgorithm


def initialization_effectiveness(
    algorithm: GeneticAlgorithm, strategies: list[str], iterations=10
):
    """Test how different population initialization strategies affect solution quality.

    Args:
        algorithm: Genetic algorithm instance.
        strategies (list): List of initialization strategy names to test.
        iterations (int): Number of test iterations for each strategy.

    Returns:
        dict: Dictionary containing fitness results for each strategy.
    """
    results = _initialization_performance_analysis(algorithm, strategies, iterations)
    plot_initialization_performance(results)
    return results


def _initialization_performance_analysis(
    algorithm: GeneticAlgorithm, strategies: list[str], iterations=10
):
    """Test different initialization strategies and collect data about solution quality."""
    results = {}
    for strategy in strategies:
        algorithm.clear_metrics()
        for _ in range(iterations):
            algorithm.population.initialize_with_strategy(strategy)
            execution_time = algorithm.evolve()

            results[strategy] = {
                "execution_time": execution_time,
                "diversity": algorithm.diversity,
                "best_fitness": algorithm.best_fitness,
                "average_fitness": algorithm.average_fitness,
                "worst_fitness": algorithm.worst_fitness,
            }
    return results


def plot_initialization_performance(results):
    """Visualize the impact of different initialization strategies on the metrics."""
    strategies = list(results.keys())
    num_generations = len(next(iter(results.values()))["diversity"])
    generations = range(num_generations)
    num_strategies = len(strategies)

    colors = plt.cm.get_cmap("tab10", num_strategies)

    # Fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(["best_fitness", "average_fitness", "worst_fitness"]):
        for j, strategy in enumerate(strategies):
            axes[i].plot(
                generations,
                results[strategy][metric],
                label=strategy.capitalize(),
                color=colors(j),
            )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True)
        axes[i].legend()

    axes[2].set_xlabel("Generation")
    fig.suptitle("Fitness Evolution Across Initialization Strategies")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    # -------------------------------------

    # Diversity plot
    plt.figure(figsize=(12, 8))
    for i, strategy in enumerate(strategies):
        plt.plot(
            generations,
            results[strategy]["diversity"],
            label=strategy.capitalize(),
            color=colors(i),
        )

    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Diversity Across Initialization Strategies")
    plt.legend()
    plt.grid(True)
    plt.show()
    # -------------------------------------

    # Execution time comparison
    plt.figure(figsize=(12, 8))
    execution_times = [results[strategy]["execution_time"] for strategy in strategies]
    plt.bar(strategies, execution_times, color=colors(range(num_strategies)))
    plt.xlabel("Initialization Strategy")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Initialization Strategy")
    plt.xticks(rotation=45)

    # -------------------------------------
    plt.show()
