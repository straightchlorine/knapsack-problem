from knapsack.analyze.utility import (
    ExperimentConfig,
    append_experiment_results,
    init_alg,
    plot_diversity,
    plot_execution_times,
    plot_optimal_generations,
    plot_performance,
    print_statistical_summary,
)
from knapsack.genetic_algorithm import GeneticAlgorithm


def mutation_impact(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config, iterations)

    plot_performance(results)
    plot_diversity(results)
    plot_execution_times(results, "Mutation rate")
    plot_optimal_generations(results, "Mutation rate")
    print_statistical_summary(results)

    return results


def _measure_metrics(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
    iterations=10,
):
    results = {}
    for mutation_rate in config.mutation_rates:
        alg.mutation_rate = mutation_rate
        key = f"Mutation rate: {mutation_rate:.2f}"

        for _ in range(iterations):
            alg.clear_metrics()
            alg.reinitialize_population()
            execution_time = alg.evolve()
            append_experiment_results(results, key, alg, execution_time)

    return results
