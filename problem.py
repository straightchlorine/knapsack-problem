from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import FitnessEvaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.elitism_selector import ElitismSelector
from knapsack.selectors.random_selector import RandomSelector

# DataInterface handles loading data from file and creating ChromosomeDataset
# objects, which contain the problem to solve

# create DataInterface object from csv file
dataset = DataInterface.from_csv("datasets/dataset.csv")

# extracts random problem from all of the generated datasets, other way would
# be to use dataset.chromosome_datasets() in order to get the whole list

# get the random problem out of the dataset
# problem = dataset.random_problem()

problem = dataset.chromosome_datasets[2]

# (pop_size, gen)
# (10, 100), very hard to obtain any good solutions especially when there is
# only 1 weight;value pair to be selected (RandomSelector)

# (10, 20), with ElitismSelector, it is possible to obtain the best solution
# very fast, even within 10 generations, that being said, while it works
# well for this solution, it restrains other possbilities very much

evaluator = FitnessEvaluator(problem)

# create genetic algorithm object
alg = GeneticAlgorithm(
    problem,
    evaluator,
    # RandomSelector(),
    ElitismSelector(evaluator),
    population_size=10,
    num_generations=10,
    mutation_rate=0.01,
)

# start the algorithm
alg.evolve()

# get the best solutions
alg.get_best_solution()
