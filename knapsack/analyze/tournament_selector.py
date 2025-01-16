import matplotlib.pyplot as plt

from knapsack.analyze.utility import (
    ExperimentConfig,
    append_experiment_results,
    init_alg,
    plot_diversity,
    plot_execution_times,
    plot_optimal_generations,
    plot_performance,
)
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.tournament_selector import TournamentSelector


def tournament_selector_analysis(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    tournament_selector: TournamentSelector,
    tournament_sizes: list[int],
    iterations=10,
):
    config.selectors = [tournament_selector]
    algorithm = init_alg(alg, config)

    results = _measure_metrics(algorithm, config, tournament_sizes, iterations)

    plot_performance(results)
    plot_diversity(results)
    plot_execution_times(results, "Generation and TournamentSelector configuration")
    plot_optimal_generations(results, "Generation and TournamentSelector configuration")

    return results


def _measure_metrics(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
    tournament_sizes: list[int],
    iterations,
):
    results = {}
    for gen in config.generations:
        for t_size in tournament_sizes:
            key = f"Gen: {gen} | Tournament size: {t_size}"
            for _ in range(iterations):
                # clear the algorithm and set parameters
                alg.clear_metrics()
                alg.generations = gen

                if hasattr(alg.selector, "tournament_size"):
                    alg.selector.tournament_size = t_size
                else:
                    raise AttributeError(
                        "Tournament selector wasn't passed or is invalid."
                    )

                alg.reinitialize_population()
                execution_time = alg.evolve()
                append_experiment_results(results, key, alg, execution_time)

    return results


def plot_tournament_selector_impact(results):
    """Visualise the impact of different parameters of TournamentSelector."""
    param_sets = list(results.keys())
    num_generations = len(next(iter(results.values()))["diversity"])
    generations = range(num_generations)
    num_sets = len(param_sets)

    colors = plt.cm.get_cmap("tab10", num_sets)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(["best_fitness", "average_fitness", "worst_fitness"]):
        for j, param_set in enumerate(param_sets):
            axes[i].plot(
                generations,
                results[param_set][metric],
                label=param_set,
                color=colors(j),
            )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True)
        axes[i].legend()

    axes[2].set_xlabel("Generation")
    fig.suptitle("Fitness Evolution Across Additional Parameters")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    # -------------------------------------

    # diversity plot
    plt.figure(figsize=(12, 8))
    for i, param_set in enumerate(param_sets):
        plt.plot(
            generations,
            results[param_set]["diversity"],
            label=param_set,
            color=colors(i),
        )

    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Diversity Across Additional Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()
    # -------------------------------------

    # execution time comparison
    plt.figure(figsize=(12, 8))
    execution_times = [results[param_set]["execution_time"] for param_set in param_sets]
    plt.bar(param_sets, execution_times, color=colors(range(num_sets)))
    plt.xlabel("Parameter Set")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Parameter Set")
    plt.xticks(rotation=45)
    plt.show()
    # -------------------------------------
