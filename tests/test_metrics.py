import matplotlib.pyplot as plt


def plot_metrics(alg):
    generations = range(len(alg.best_fitness))

    plt.figure(figsize=(15, 10))

    # Plot Best Fitness
    plt.subplot(3, 1, 1)
    plt.plot(generations, alg.best_fitness, label="Best Fitness", color="blue")
    plt.title("Best Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()

    # Plot Average Fitness
    plt.subplot(3, 1, 2)
    plt.plot(
        generations,
        alg.average_fitness,
        label="Average Fitness",
        color="green",
    )
    plt.title("Average Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()

    # Plot Genetic Diversity
    plt.subplot(3, 1, 3)
    plt.plot(generations, alg.diversity, label="Diversity (%)", color="red")
    plt.title("Genetic Diversity Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Diversity (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
