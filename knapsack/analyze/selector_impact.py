import matplotlib.pyplot as plt

from knapsack.analyze.utility import (
    ExperimentConfig,
    ExperimentResults,
    init_alg,
    is_class_equal,
)
from knapsack.genetic_algorithm import GeneticAlgorithm


def selector_impact_analysis(alg: type[GeneticAlgorithm], config: ExperimentConfig):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config)
    plot_selector_impact_metrics(results)


def _measure_metrics(alg: GeneticAlgorithm, config: ExperimentConfig):
    results = {}
    for selector in config.selectors:
        alg.clear_metrics()

        if not is_class_equal(alg.selector, selector):
            alg.selector = selector

        execution_time = alg.evolve()
        alg.reinitialize_population()

        key = type(selector).__name__
        results[key] = ExperimentResults(
            metadata={
                "population_size": alg.population_size,
                "mutation_rate": alg.mutation_rate,
                "selector": type(selector).__name__,
                "operator": type(alg.crossover_operator).__name__,
                "evaluator": type(alg.evaluator).__name__,
                "generations": alg.generations,
            },
            execution_time=execution_time,
            diversity=alg.diversity,
            best_fitness=alg.best_fitness,
            average_fitness=alg.average_fitness,
            worst_fitness=alg.worst_fitness,
        )

    return results


def plot_selector_impact_metrics(results):
    """Visualise impact of different selectors on the metrics."""
    selectors = list(results.keys())
    num_generations = len(next(iter(results.values()))["diversity"])
    generations = range(num_generations)
    num_selectors = len(selectors)

    colors = plt.cm.get_cmap("tab10", num_selectors)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(["best_fitness", "average_fitness", "worst_fitness"]):
        for j, selector in enumerate(selectors):
            axes[i].plot(
                generations,
                results[selector][metric],
                label=selector,
                color=colors(j),
            )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True)
        axes[i].legend()

    axes[2].set_xlabel("Generation")
    fig.suptitle("Fitness Evolution Across Selection Methods")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    # -------------------------------------

    # diversity plot
    plt.figure(figsize=(12, 8))
    for i, selector in enumerate(selectors):
        plt.plot(
            generations,
            results[selector]["diversity"],
            label=selector,
            color=colors(i),
        )

    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Diversity Across Selection Methods")
    plt.legend()
    plt.grid(True)
    plt.show()
    # -------------------------------------

    # execution time comparison
    plt.figure(figsize=(12, 8))
    execution_times = [results[selector]["execution_time"] for selector in selectors]
    plt.bar(selectors, execution_times, color=colors(range(num_selectors)))
    plt.xlabel("Selection Method")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Selection Method")
    plt.xticks(rotation=45)

    # -------------------------------------
    plt.show()
