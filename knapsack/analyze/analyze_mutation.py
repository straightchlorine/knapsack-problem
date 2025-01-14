import matplotlib.pyplot as plt


def mutation_analysis(alg, mutation_rates):
    results = analyze_mutation_probability(alg, mutation_rates)
    plot_mutation_analysis(results)


def analyze_mutation_probability(alg, mutation_rates):
    results = {}
    for rate in mutation_rates:
        alg.mutation_rate = rate
        execution_time = alg.evolve()

        results[rate] = {
            "execution_time": execution_time,
            "diversity": alg.diversity,
            "best_fitness": alg.best_fitness,
            "average_fitness": alg.average_fitness,
            "worst_fitness": alg.worst_fitness,
        }

    return results


def plot_mutation_analysis(results):
    """
    Visualize the results of different mutation probabilities.

    Args:
        results (dict): Dictionary containing results for each mutation probability
    """
    mutation_rates = list(results.keys())
    num_generations = len(next(iter(results.values()))["diversity"])
    generations = range(num_generations)
    num_rates = len(mutation_rates)

    colors = plt.cm.get_cmap("tab10", num_rates)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(
        ["best_fitness", "average_fitness", "worst_fitness"]
    ):
        for j, rate in enumerate(mutation_rates):
            axes[i].plot(
                generations,
                results[rate][metric],
                label=str(rate),
                color=colors(j),
            )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True)
        axes[i].legend()

    axes[2].set_xlabel("Generation")
    fig.suptitle("Fitness Evolution Across Mutation Probabilities")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    # diversity plot
    plt.figure(figsize=(12, 8))
    for i, rate in enumerate(mutation_rates):
        plt.plot(
            generations,
            results[rate]["diversity"],
            label=str(rate),
            color=colors(i),
        )

    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Diversity Across Mutation Probabilities")
    plt.legend()
    plt.grid(True)
    plt.show()

    # execution time comparison
    plt.figure(figsize=(12, 8))
    execution_times = [
        results[rate]["execution_time"] for rate in mutation_rates
    ]
    plt.bar(mutation_rates, execution_times, color=colors(range(num_rates)))
    plt.xlabel("Mutation Probability")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Mutation Probability")
    plt.xticks(rotation=45)
    plt.show()
