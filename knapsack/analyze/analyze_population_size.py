import matplotlib.pyplot as plt


def population_analysis(
    alg_class,
    problem,
    evaluator,
    selector,
    crossover_operator,
    mutation_rate,
    generations,
    population_sizes,
):
    results = analyze_population_size(
        alg_class,
        problem,
        evaluator,
        selector,
        crossover_operator,
        mutation_rate,
        generations,
        population_sizes,
    )
    plot_population_analysis(results)


def analyze_population_size(
    alg_class,
    problem,
    evaluator,
    selector,
    crossover,
    mutation_rate,
    num_generations,
    population_sizes,
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


def plot_population_analysis(results):
    """
    Visualize the results from the evolutionary algorithm analysis with improved clarity.

    Args:
        results (dict): Dictionary containing results for each population size
    """
    population_sizes = list(results.keys())
    num_generations = len(results[population_sizes[0]]["diversity"])
    generations = range(num_generations)
    num_pop_sizes = len(population_sizes)

    colors = plt.cm.get_cmap("tab10", num_pop_sizes)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(
        ["best_fitness", "average_fitness", "worst_fitness"]
    ):
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

    # execution time plot
    plt.figure(figsize=(12, 8))
    execution_times = [
        results[size]["execution_time"] for size in population_sizes
    ]
    plt.bar(
        population_sizes, execution_times, color=colors(range(num_pop_sizes))
    )
    plt.xlabel("Population Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Population Size")
    plt.show()
