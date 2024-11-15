import numpy as np

from knapsack.population import Population
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class GeneticAlgorithm:
    def __init__(
        self,
        dataset,
        evaluator: Evaluator,
        selector: Selector,
        population_size=100,
        num_generations=500,
        mutation_rate=0.01,
    ):
        self.dataset = dataset
        self.gene_length = self.dataset.length

        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        # set the evaluator and selector objects
        self.evaluator = evaluator
        self.selector = selector

        # create and initialize the first population
        self.population = Population(
            self.selector, self.population_size, self.gene_length
        )
        self.population.initialize()

    def crossover(self, parent1, parent2):
        # choose random point for crossover
        point = np.random.randint(1, parent1.size - 1)

        # generate genes for two children
        genes = (
            np.concatenate([parent1[:point], parent2[point:]]),
            np.concatenate([parent1[point:], parent2[:point]]),
        )

        # return two chromosomes
        return np.array([genes[0], genes[1]])

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
            children = self.crossover(parent1, parent2)

            # mutate the children
            for child in children:
                child = self.mutate(child)

            # add the child to the new population
            new_population.add_chromosome(children)

        return new_population

    def evolve(self):
        """Evolve the population for a set number of generations."""
        for _ in range(self.num_generations):
            self.population = self.new_generation()
            self.population.update_selector()

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

        for evaluated in ordered:
            self.prompt(evaluated)

        return ordered[0]
