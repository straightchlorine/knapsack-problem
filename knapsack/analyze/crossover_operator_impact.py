import matplotlib.pyplot as plt

from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.operators.crossover import Crossover


def crossover_operator_impact_analysis(
    alg: GeneticAlgorithm, crossover_operators: list[Crossover]
):
    """Analyse the impact of different crossover operators on the metrics.

    Args:
        alg (GeneticAlgorithm): Genetic algorithm instance.
        crossover_operators (list): List of crossover operators to test.
    """
    results = _measure_metrics(alg, crossover_operators)
    plot_crossover_impact_metrics(results)


def _measure_metrics(alg: GeneticAlgorithm, crossover_operators: list[Crossover]):
    results = {}
    for crossover_operator in crossover_operators:
        alg.crossover_operator = crossover_operator
        execution_time = alg.evolve()

        results[type(crossover_operator).__name__] = {
            "execution_time": execution_time,
            "diversity": alg.diversity,
            "best_fitness": alg.best_fitness,
            "average_fitness": alg.average_fitness,
            "worst_fitness": alg.worst_fitness,
        }

    return results


def plot_crossover_impact_metrics(results):
    """Visualize the impact of different crossover operators on the metrics."""
    crossover_operators = list(results.keys())
    num_generations = len(next(iter(results.values()))["diversity"])
    generations = range(num_generations)
    num_operators = len(crossover_operators)

    colors = plt.cm.get_cmap("tab10", num_operators)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(["best_fitness", "average_fitness", "worst_fitness"]):
        for j, operator in enumerate(crossover_operators):
            axes[i].plot(
                generations,
                results[operator][metric],
                label=operator,
                color=colors(j),
            )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True)
        axes[i].legend()

    axes[2].set_xlabel("Generation")
    fig.suptitle("Fitness Evolution Across Crossover Methods")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    # -------------------------------------

    # diversity plot
    plt.figure(figsize=(12, 8))
    for i, operator in enumerate(crossover_operators):
        plt.plot(
            generations,
            results[operator]["diversity"],
            label=operator,
            color=colors(i),
        )

    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Diversity Across Crossover Methods")
    plt.legend()
    plt.grid(True)
    plt.show()
    # -------------------------------------

    # execution time comparison
    plt.figure(figsize=(12, 8))
    execution_times = [
        results[operator]["execution_time"] for operator in crossover_operators
    ]
    plt.bar(
        crossover_operators,
        execution_times,
        color=colors(range(num_operators)),
    )
    plt.xlabel("Crossover Method")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Crossover Method")
    plt.xticks(rotation=45)
    plt.show()
    # -------------------------------------
