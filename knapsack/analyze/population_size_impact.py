import matplotlib.pyplot as plt

from knapsack.dataset import Dataset
from knapsack.evaluators.evaluator import Evaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.operators.crossover import Crossover
from knapsack.selectors.selector import Selector


def population_impact_analysis(
    alg_class: type[GeneticAlgorithm],
    problem: Dataset,
    evaluator: Evaluator,
    selector: Selector,
    crossover_operator: Crossover,
    mutation_rate: float,
    generations: int,
    population_sizes: list[int],
):
    """Measure and plot impact of population size on the metrics.

    Args:
        alg_class (class): Genetic algorithm class.
        problem (Chromosome): Problem to solve.
        evaluator (Evaluator): Evaluator object.
        selector (Selector): Selector object.
        crossover_operator (Crossover): Crossover operator.
        mutation_rate (float): Mutation rate.
        generations (int): Number of generations.
        population_sizes (list): Population sizes to test.
    """
    results = _measure_metrics(
        alg_class,
        problem,
        evaluator,
        selector,
        crossover_operator,
        mutation_rate,
        generations,
        population_sizes,
    )
    plot_population_impact_metrics(results)


def _measure_metrics(
    alg_class: type[GeneticAlgorithm],
    problem: Dataset,
    evaluator: Evaluator,
    selector: Selector,
    crossover: Crossover,
    mutation_rate: float,
    num_generations: int,
    population_sizes: list[int],
):
    results = {}
    for size in population_sizes:
        alg = alg_class(
            problem,
            evaluator,
            selector,
            crossover,
            population_size=size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
        )
        execution_time = alg.evolve()

        results[size] = {
            "execution_time": execution_time,
            "diversity": alg.diversity,
            "best_fitness": alg.best_fitness,
            "average_fitness": alg.average_fitness,
            "worst_fitness": alg.worst_fitness,
        }
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
