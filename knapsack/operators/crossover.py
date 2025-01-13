class Crossover:
    """Base class for crossover operators."""

    def __init__(self, dev=False):
        self.dev = dev

    def __str__(self):
        return f"Crossover operator: {self.__class__.__name__}\n{35*'='}"

    def crossover(self, parent_a, parent_b):
        """Perform crossover in order to generate offspring.

        Args:
            parent_a (np.ndarray): First parent.
            parent_b (np.ndarray): Second parent.

        Returns:
            np.ndarray: Children from the crossover operation.
        """
        raise NotImplementedError(
            "Method 'crossover' must be implemented in a subclass."
        )
