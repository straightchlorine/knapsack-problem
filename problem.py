from knapsack.dataset import DataInterface
from knapsack.evaluators.fitness import FitnessEvaluator
from knapsack.genetic_algorithm import GeneticAlgorithm
from knapsack.selectors.random_selector import RandomSelector

# DataInterface handles loading data from file and creating ChromosomeDataset
# objects, which contain the problem to solve

# create DataInterface object from csv file
dataset = DataInterface.from_csv("datasets/dataset.csv")

# extracts random problem from all of the generated datasets, other way would
# be to use dataset.chromosome_datasets() in order to get the whole list

# get the random problem out of the dataset
problem = dataset.random_problem()

# create genetic algorithm object
alg = GeneticAlgorithm(
    problem,
    FitnessEvaluator(problem),
    RandomSelector(),
    population_size=10,
    num_generations=100,
    mutation_rate=0.01,
)

# start the algorithm
alg.evolve()

# get the best solutions
alg.get_best_solution()
