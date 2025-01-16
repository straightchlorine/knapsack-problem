from itertools import product

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
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.mutations.mutation import Mutation
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


def parameters_impact_analysis(
    alg: type[GeneticAlgorithm],
    config: ExperimentConfig,
    iterations=10,
):
    algorithm = init_alg(alg, config)
    results = _measure_metrics(algorithm, config, iterations)

    label = "Configuration"
    plot_performance(results)
    plot_diversity(results)
    plot_execution_times(results, label)
    plot_optimal_generations(results, label)
    print_statistical_summary(results)

    return results


def _generate_config_key(
    population_size: int,
    mutation_rate: float,
    generations: int,
    evaluator: Evaluator,
    selector: Selector,
    crossover_operator: Crossover,
    mutation_operator: Mutation,
    id: int,
) -> str:
    """Generate a unique key for a parameter configuration."""
    pop = f"Population: {population_size}"
    gen = f"Generations: {generations}"
    eval = f"Evaluator: {type(evaluator).__name__}"
    mut = f"Mutation rate: {mutation_rate}"
    mut_op = f"Mutation operator: {type(mutation_operator).__name__}"
    sel = f"Selector: {type(selector).__name__}"
    cross = f"Crossover: {type(crossover_operator).__name__}"

    config = f"Configuration ID{id}:\n\t{pop}\n\t{gen}\n\t{eval}\n\t{mut}\n\t{mut_op}\n\t{sel}\n\t{cross}"
    print(config)
    return f"Configuration ID: {id}"


def _measure_metrics(
    alg: GeneticAlgorithm,
    config: ExperimentConfig,
    iterations,
):
    results = {}
    id = 0

    # iterate over all combinations of parameters
    for params in product(
        config.evaluators,
        config.selectors,
        config.crossover_operators,
        config.mutation_operators,
        config.mutation_rates,
        config.population_sizes,
        config.generations,
        config.strategies,
    ):
        (
            evaluator,
            selector,
            crossover_operator,
            mutation_operator,
            mutation_rate,
            population_size,
            gens,
            strategy,
        ) = params

        id += 1
        key = _generate_config_key(
            population_size,
            mutation_rate,
            gens,
            evaluator,
            selector,
            crossover_operator,
            mutation_operator,
            id,
        )

        for _ in range(iterations):
            # set all the parameters for the run and reinitialize population
            alg.evaluator = evaluator
            alg.selector = selector
            alg.crossover_operator = crossover_operator
            alg.mutation_operator = mutation_operator
            alg.mutation_rate = mutation_rate
            alg.population_size = population_size
            alg.generations = gens
            alg.strategy = strategy
            alg.reinitialize_population()

            execution_time = alg.evolve()

            append_experiment_results(results, key, alg, execution_time)
    return results
