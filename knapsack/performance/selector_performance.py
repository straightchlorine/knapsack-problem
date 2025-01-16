
import matplotlib.pyplot as plt
import numpy as np

from knapsack.analyze.utility import (
    ExperimentConfig,
    ExperimentResults,
    append_experiment_results,
    init_alg,
    print_statistical_summary,
)
from knapsack.genetic_algorithm import GeneticAlgorithm


def selector_effectiveness(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    """Test and visualise performance of various selection methods.

    Args:
        alg (type[GeneticAlgorithm]): Genetic algorithm class.
        config (ExperimentConfig): Experiment configuration.
        iterations (int, optional): Number of iterations. Defaults to 10.
    """
    algorithm = init_alg(alg, config)
    results = _selection_performance_analysis(algorithm, config, iterations)

    plot_selector_performance(results)
    plot_diversity(results)
    plot_execution_times(results)
    plot_optimal_generations(results)
    print_statistical_summary(results)

    return results


def _selection_performance_analysis(
    alg: GeneticAlgorithm, config: ExperimentConfig, iterations=10
):
    results = {}
    for selector in config.selectors:
        alg.selector = selector
        key = type(selector).__name__

        for _ in range(iterations):
            alg.clear_metrics()
            alg.reinitialize_population()
            execution_time = alg.evolve()
            append_experiment_results(results, key, alg, execution_time)

    return results


def plot_selector_performance(results: dict[str, ExperimentResults]):
    """
    Plot the comparison of selection methods for fitness metrics (best, average, worst)
    with standard deviation bands.
    Args:
        results (dict): Dictionary of ExperimentResults objects containing fitness metrics.
    """
    # Setup for plotting
    selector_operators = list(results.keys())
    colors = plt.cm.get_cmap("tab10", len(selector_operators))
    metrics = ["best_fitness", "average_fitness", "worst_fitness"]
    titles = ["Best Fitness", "Average Fitness", "Worst Fitness"]

    # Create figure and subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for j, operator in enumerate(selector_operators):
            # Get data
            generations = range(1, results[operator].metadata["generations"] + 1)
            mean_data = np.array(getattr(results[operator], f"mean_{metric}"))
            std_data = np.std(
                [getattr(results[operator], f"mean_{metric}") for _ in range(5)], axis=0
            )

            # Plot mean line
            axes[i].plot(
                generations,
                mean_data,
                label=operator,
                color=colors(j),
                marker="o",
                markersize=4,
            )

            # Add standard deviation bands
            axes[i].fill_between(
                generations,
                mean_data - std_data,
                mean_data + std_data,
                alpha=0.2,
                color=colors(j),
            )

            # Customize plot
            axes[i].set_title(f"{title} Over Generations")
            axes[i].set_ylabel(title)
            axes[i].grid(True)

            # Add legend to the first subplot only
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set common x-label
    axes[-1].set_xlabel("Generation")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_diversity(results: dict[str, ExperimentResults]):
    """
    Plot the population diversity comparison for different selection methods.
    Args:
        results (dict): Dictionary of ExperimentResults objects containing diversity metrics.
    """
    plt.figure(figsize=(12, 6))
    selector_operators = list(results.keys())
    colors = plt.cm.get_cmap("tab10", len(selector_operators))

    for i, operator in enumerate(selector_operators):
        if results[operator].diversity:  # Check if diversity data exists
            generations = range(1, results[operator].metadata["generations"] + 1)
            mean_data = np.array(results[operator].mean_diversity)
            std_data = np.std(
                [results[operator].mean_diversity for _ in range(5)], axis=0
            )

            plt.plot(
                generations,
                mean_data,
                label=operator,
                color=colors(i),
                marker="o",
                markersize=4,
            )

            plt.fill_between(
                generations,
                mean_data - std_data,
                mean_data + std_data,
                alpha=0.2,
                color=colors(i),
            )

    plt.title("Population Diversity Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_execution_times(results: dict[str, ExperimentResults]):
    """
    Plot the execution times comparison for different selection methods.

    Args:
        results (dict): Dictionary of ExperimentResults objects containing execution times.
    """
    plt.figure(figsize=(10, 6))
    selector_operators = list(results.keys())

    execution_times = [
        results[operator].mean_execution_time for operator in selector_operators
    ]

    bars = plt.bar(selector_operators, execution_times)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}ms",
            ha="center",
            va="bottom",
        )

    plt.title("Execution Time Comparison")
    plt.xlabel("Selection Operator")
    plt.ylabel("Execution Time (seconds)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_optimal_generations(results: dict[str, ExperimentResults]):
    """
    Plot a comparison of optimal generations (when best solution was found) for different selectors.

    Args:
        results (dict): Dictionary of ExperimentResults objects containing optimal generation data.
    """
    plt.figure(figsize=(10, 6))
    selector_operators = list(results.keys())

    optimal_gens = [
        results[operator].mean_optimal_generation for operator in selector_operators
    ]

    # Create bars with different colors
    bars = plt.bar(selector_operators, optimal_gens)
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"Gen {int(height)}",
            ha="center",
            va="bottom",
        )

    plt.title("Optimal Solution Discovery Speed Comparison")
    plt.xlabel("Selection Method")
    plt.ylabel("Generation When Best Solution Found")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    avg_gen = float(np.mean(optimal_gens))
    plt.axhline(
        y=avg_gen, color="r", linestyle="--", label=f"Average (Gen {avg_gen:.1f})"
    )
    plt.legend()

    plt.tight_layout()
    plt.show()
