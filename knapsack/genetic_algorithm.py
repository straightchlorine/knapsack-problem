import numpy as np

from knapsack.base.chromosome import Chromosome
from knapsack.base.population import Population
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
        self.evaluator = evaluator

        self.gene_length = self.dataset.length

        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        self.selector = selector
        self.population = Population(
            self.selector, self.population_size, self.gene_length
        )

    def crossover(self, parent1, parent2):
        # choose random point for crossover
        point = np.random.randint(1, parent1.genes.size - 1)

        # child genes belong to parent1 before the point and to parent2 after
        child_genes = np.concatenate(
            [parent1.genes[:point], parent2.genes[point:]]
        )

        # return a new chromosome with the child genes
        return Chromosome(child_genes)

        # fix
        # # child genes belong to parent1 before the point and to parent2 after
        # child_genes = np.concatenate(
        #     [parent1.genes[:point], parent2.genes[point:]]
        # )
        #
        # child_genes = np.concatenate(
        #     [parent1.genes[point:], parent2.genes[:point]]
        # )
        #
        # # return a new chromosome with the child genes
        # return (Chromosome(child_genes), Chromosome(child_genes))

    def mutate(self, chromosome):
        # iterate through the genes
        for i in range(chromosome.length):
            # if randomly selected probability is less than set, flip the genes
            if np.random.rand() < self.mutation_rate:
                chromosome.genes[i] = 1 - chromosome.genes[i]  # Flip the gene
        return chromosome

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
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)

            # add later fix
            # add the child to the new population
            new_population.add_chromosome(child)

        return new_population

    def evolve(self):
        for _ in range(self.num_generations):
            self.population = self.new_generation()

    def prompt(self, evaluated):
        total_prompt = "\n==============================\n"
        total_prompt += str(evaluated)
        total_prompt += "\n==============================\n"
        total_prompt += "total weight: "
        total_prompt += str(np.sum(evaluated.genes * self.dataset.weights))
        total_prompt += " | "
        total_prompt += "capacity: "
        total_prompt += str(float(self.dataset.capacity))
        total_prompt += "\n"
        total_prompt += "total value: "
        total_prompt += str(np.sum(evaluated.genes * self.dataset.values))

        print(total_prompt)

    def get_best_solution(self, n=1):
        ordered = sorted(
            self.population.chromosomes,
            key=lambda chrom: self.evaluator.evaluate(chrom),
            reverse=True,
        )[:n]

        print(f"\n{self.dataset}")

        for evaluated in ordered:
            self.prompt(evaluated)

        return ordered[0]
