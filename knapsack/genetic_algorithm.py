import time

import numpy as np

from knapsack.evaluators.evaluator import Evaluator
from knapsack.operators.crossover import Crossover
from knapsack.population import Population
from knapsack.selectors.selector import Selector


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000


class GeneticAlgorithm:
    def __init__(
        self,
        dataset,
        evaluator: Evaluator,
        selector: Selector,
        operator: Crossover,
        population_size=100,
        num_generations=500,
        mutation_rate=0.01,
    ):
        self.dev = False
        self.dataset = dataset
        self.gene_length = self.dataset.length

        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        # set the evaluator and selector objects
        self.evaluator = evaluator
        print(evaluator)

        self.selector = selector
        print(selector)

        self.operator = operator
        print(operator)

        # create and initialize the first population
        self.population = Population(
            self.selector, self.population_size, self.gene_length
        )
        self.population.initialize()

    def mutate(self, genes):
        # iterate through the genes
        for i in genes:
            # if randomly selected probability is less than set, flip the genes
            if np.random.rand() < self.mutation_rate:
                genes[i] = 1 - genes[i]
        return genes

    def new_generation(self):
        # new generation
        new_population = Population(
            self.selector, self.population_size, self.gene_length
        )

        while len(new_population.chromosomes) < self.population_size:
            parents = self.population.select_parents()
            # chose two parents from the selected chromosomes
            parent1, parent2 = parents

            # create a child by crossover and mutation
            children = self.operator.crossover(parent1, parent2)

            # mutate the children
            for child in children:
                child = self.mutate(child)

            # add the child to the new population
            new_population.add_chromosome(children)

        return new_population

    def evolve(self):
        """Evolve the population for a set number of generations."""
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []

        with Timer() as timer:
            for generation in range(self.num_generations):
                best_solution = self.get_best_solution()
                best_fitness = self.get_solution_fitness(best_solution)
                avg_fitness = np.mean(
                    [
                        self.get_solution_fitness(chrom)
                        for chrom in self.population.chromosomes
                    ]
                )
                worst_fitness = np.min(
                    [
                        self.get_solution_fitness(chrom)
                        for chrom in self.population.chromosomes
                    ]
                )

                diversity = self.population.measure_diversity()

                self.best_fitness.append(best_fitness)
                self.average_fitness.append(avg_fitness)
                self.worst_fitness.append(worst_fitness)
                self.diversity.append(diversity)

                if self.dev:
                    print(
                        f"Generation {generation}: Best Fitness: {best_fitness:.2f}, "
                        f"Average Fitness: {avg_fitness:.2f}, Diversity: {diversity:.2f}%"
                    )

                self.population = self.new_generation()
                self.population.update_selector()

        if self.dev:
            print("=" * 35)
            print(f"Evolution took {timer.interval:.4f} miliseconds.")
        return timer.interval

    def prompt(self, evaluated):
        """Prompt the evaluated chromosome.

        Args:
            evaluated (np.ndarray): Evaluated chromosome.
        """
        total_weight = np.sum(evaluated * self.dataset.weights)
        total_value = np.sum(evaluated * self.dataset.values)
        evaluation = self.evaluator.evaluate(evaluated)

        capacity = float(self.dataset.capacity)

        total_prompt = "\n".join(
            [
                "=" * 30,
                f"genes={evaluated}",
                "=" * 30,
                f"total weight: {total_weight} | capacity: {capacity}",
                f"total value: {total_value} | evaluation: {evaluation}",
            ]
        )
        print(total_prompt)

    def get_solution_fitness(self, solution):
        """Get the fitness of the solution.
        Args:
            solution (np.ndarray): Solution to evaluate.
        Returns:
            float: Fitness of the solution.
        """
        return self.evaluator.evaluate(solution)

    def get_population_statistics(self):
        """Gather fitness statistics for the current population."""
        fitness_scores = [
            self.evaluator.evaluate(chrom)
            for chrom in self.population.chromosomes
        ]
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = min(fitness_scores)

        return {
            "best_fitness": best_fitness,
            "average_fitness": avg_fitness,
            "worst_fitness": worst_fitness,
        }

    def get_best_solution(self, n=1):
        """Get the best solutions from the population.

        Args:
            n (int): Display top n solutions from the population.

        Returns:
            np.ndarray: Best solution from the population.
        """
        ordered = sorted(
            self.population.chromosomes,
            key=lambda chrom: self.evaluator.evaluate(chrom),
            reverse=True,
        )[:n]

        if self.dev:
            for evaluated in ordered:
                self.prompt(evaluated)

        return ordered[0]
