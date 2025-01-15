class Evaluator:
    """Base class for chromosome evaluators."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __str__(self):
        return f"Evaluator: {self.__class__.__name__} | Dataset: {self.dataset}"

    def evaluate(self, chromosome):
        """Evaluate the chromosome."""
        raise NotImplementedError("Method 'evaluate' must be implemented in a subclass.")
