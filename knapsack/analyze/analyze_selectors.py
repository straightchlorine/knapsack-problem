import matplotlib.pyplot as plt


def selection_analysis(alg, selectors):
    results = analyze_selection_methods(alg, selectors)
    plot_selection_analysis(results)


def analyze_selection_methods(alg, selectors):
    results = {}
    for selector in selectors:
        alg.selector = selector
        execution_time = alg.evolve()

        results[type(selector).__name__] = {
            "execution_time": execution_time,
            "diversity": alg.diversity,
            "best_fitness": alg.best_fitness,
            "average_fitness": alg.average_fitness,
            "worst_fitness": alg.worst_fitness,
        }

    return results


def plot_selection_analysis(results):
    """
    Visualize the results of different selection methods.

    Args:
        results (dict): Dictionary containing results for each selection method
    """
    selectors = list(results.keys())
    num_generations = len(next(iter(results.values()))["diversity"])
    generations = range(num_generations)
    num_selectors = len(selectors)

    colors = plt.cm.get_cmap("tab10", num_selectors)

    # fitness during evolution by generation
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, metric in enumerate(
        ["best_fitness", "average_fitness", "worst_fitness"]
    ):
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

    # execution time comparison
    plt.figure(figsize=(12, 8))
    execution_times = [
        results[selector]["execution_time"] for selector in selectors
    ]
    plt.bar(selectors, execution_times, color=colors(range(num_selectors)))
    plt.xlabel("Selection Method")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Selection Method")
    plt.xticks(rotation=45)
    plt.show()
