from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence
import matplotlib.pyplot as plt
import numpy as np
from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


@dataclass
class ExperimentConfig:
    """Configuration parameters for genetic algorithm experiments."""

    problems: list[Dataset]
    evaluators: Sequence[Evaluator]
    selectors: Sequence[Selector]
    crossover_operators: Sequence[Crossover]
    mutation_operators: Sequence[Mutation]
    population_sizes: list[int]
    generations: list[int]
    mutation_rates: list[float]
    strategies: list[str]


@dataclass
class ExperimentResults:
    """Results from a single genetic algorithm run."""

    metadata: dict[str, Any]
    execution_time: list[float]
    diversity: list[list[float]]
    best_fitness: list[list[float]]
    average_fitness: list[list[float]]
    worst_fitness: list[list[float]]
    optimal_generation: list[int]

    @property
    def mean_execution_time(self) -> float:
        return fmean(self.execution_time) if self.execution_time else 0.0

    @property
    def mean_optimal_generation(self) -> float:
        return fmean(self.optimal_generation) if self.optimal_generation else 0.0

    @property
    def mean_diversity(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.diversity)]
        return means

    @property
    def mean_best_fitness(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.best_fitness)]
        return means

    @property
    def mean_average_fitness(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.average_fitness)]
        return means

    @property
    def mean_worst_fitness(self) -> list[float]:
        means = [fmean(values) for values in zip(*self.worst_fitness)]
        return means


def init_alg(alg: type[GeneticAlgorithm], config: ExperimentConfig):
    algorithm = alg(
        config.problems[0],
        config.evaluators[0],
        config.selectors[0],
        config.crossover_operators[0],
        config.mutation_operators[0],
        config.population_sizes[0],
        config.generations[0],
        strategy=config.strategies[0],
    )
    algorithm.dev = False
    return algorithm


def is_class_equal(obj1, obj2):
    """Check if a property is equal between two objects."""
    return obj1.__class__.__name__ == obj2.__class__.__name__


def append_experiment_results(results: dict, key, alg: GeneticAlgorithm, execution_time):
    if key not in results:
        results[key] = ExperimentResults(
            metadata={
                "population_size": alg.population_size,
                "mutation_rate": alg.mutation_rate,
                "selector": type(alg.selector).__name__,
                "operator": type(alg.crossover_operator).__name__,
                "evaluator": type(alg.evaluator).__name__,
                "generations": alg.generations,
            },
            execution_time=[execution_time],
            diversity=[],
            best_fitness=[],
            average_fitness=[],
            worst_fitness=[],
            optimal_generation=[],
        )

        results[key].diversity.append(alg.diversity)
        results[key].best_fitness.append(alg.best_fitness)
        results[key].average_fitness.append(alg.average_fitness)
        results[key].worst_fitness.append(alg.worst_fitness)
        results[key].optimal_generation.append(alg.optimal_generation)

    else:
        results[key].execution_time.append(execution_time)
        results[key].diversity.append(alg.diversity)
        results[key].best_fitness.append(alg.best_fitness)
        results[key].average_fitness.append(alg.average_fitness)
        results[key].worst_fitness.append(alg.worst_fitness)
        results[key].optimal_generation.append(alg.optimal_generation)


def print_statistical_summary(results: dict[str, ExperimentResults]):
    """
    Print statistical summary of the results.

    Args:
        results (dict): Dictionary of ExperimentResults objects.
    """
    print("\nStatistical Summary:")
    for operator, data in results.items():
        print(f"\n{operator}:")
        print(f"Execution Time: {data.mean_execution_time:.2f} miliseconds")
        print(f"Best Fitness: {fmean(data.mean_best_fitness):.4f}")
        print(f"Average Fitness: {fmean(data.mean_average_fitness):.4f}")
        print(f"Worst Fitness: {fmean(data.mean_worst_fitness):.4f}")
        print(f"Population Diversity: {fmean(data.mean_diversity):.4f}")
        print(f"Optimal Generation: {data.mean_optimal_generation}")


def plot_performance(results: dict[str, ExperimentResults]):
    """
    Plot the comparison of fitness metrics (best, average, worst)
    with standard deviation bands.
    Args:
        results (dict): Dictionary of ExperimentResults objects containing fitness metrics.
    """
    # Setup for plotting
    keys = list(results.keys())
    colors = plt.cm.get_cmap("tab10", len(keys))
    metrics = ["best_fitness", "average_fitness", "worst_fitness"]
    titles = ["Best Fitness", "Average Fitness", "Worst Fitness"]

    # Create figure and subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for j, operator in enumerate(keys):
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
    key = list(results.keys())
    colors = plt.cm.get_cmap("tab10", len(key))

    for i, operator in enumerate(key):
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


def plot_execution_times(results: dict[str, ExperimentResults], label):
    """
    Plot the execution times comparison for different selection methods.

    Args:
        results (dict): Dictionary of ExperimentResults objects containing execution times.
    """
    plt.figure(figsize=(10, 6))
    key = list(results.keys())

    execution_times = [results[operator].mean_execution_time for operator in key]

    bars = plt.bar(key, execution_times)

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
    plt.xlabel(label)
    plt.ylabel("Execution Time (seconds)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_optimal_generations(results: dict[str, ExperimentResults], label):
    """
    Plot a comparison of optimal generations (when best solution was found) for different selectors.

    Args:
        results (dict): Dictionary of ExperimentResults objects containing optimal generation data.
    """
    plt.figure(figsize=(10, 6))
    key = list(results.keys())

    optimal_gens = [results[operator].mean_optimal_generation for operator in key]

    # Create bars with different colors
    bars = plt.bar(key, optimal_gens)
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
    plt.xlabel(label)
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
