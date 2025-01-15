import matplotlib.pyplot as plt

from knapsack.analyze.utility import ExperimentConfig, ExperimentResults, init_alg
from knapsack.genetic_algorithm import GeneticAlgorithm


def population_impact_analysis(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config)
    plot_population_impact_metrics(results)


def _measure_metrics(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
):
    results = {}
    for size in config.population_sizes:
        alg.clear_metrics()
        if alg.population_size != size:
            alg.population_size = size
            alg.reinitialize_population()

        execution_time = alg.evolve()

        key = size
        results[key] = ExperimentResults(
            metadata={
                "population_size": alg.population_size,
                "mutation_rate": alg.mutation_rate,
                "selector": type(alg.selector).__name__,
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


def plot_population_impact_metrics(results):
    """Visualize the population size's impact on metrics."""
    population_sizes = list(results.keys())
    num_generations = len(results[population_sizes[0]]["diversity"])
    generations = range(num_generations)
    num_pop_sizes = len(population_sizes)

    colors = plt.cm.get_cmap("tab10", num_pop_sizes)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(["best_fitness", "average_fitness", "worst_fitness"]):
        for j, pop_size in enumerate(population_sizes):
            axes[i].plot(
                generations,
                results[pop_size][metric],
                label=f"Pop={pop_size}",
                color=colors(j),
            )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True)
        axes[i].legend()

    axes[2].set_xlabel("Generation")
    fig.suptitle("Fitness Evolution Over Generations")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    # -------------------------------------

    # diversity plot by generation
    plt.figure(figsize=(12, 8))
    for i, pop_size in enumerate(population_sizes):
        plt.plot(
            generations,
            results[pop_size]["diversity"],
            label=f"Pop={pop_size}",
            color=colors(i),
        )

    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Population Diversity Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()
    # -------------------------------------

    # execution time plot
    plt.figure(figsize=(12, 8))
    execution_times = [results[size]["execution_time"] for size in population_sizes]
    plt.bar(population_sizes, execution_times, color=colors(range(num_pop_sizes)))
    plt.xlabel("Population Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Population Size")
    # -------------------------------------

    plt.show()
