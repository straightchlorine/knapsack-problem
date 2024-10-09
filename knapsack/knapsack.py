#!/usr/bin/env python

import numpy as np
import pandas as pd


class KnapsackProblem:
    def __init__(self, weights, values, capacity):
        """Main class for the knapsack problem.

        Fields:
            weights (np.ndarray): Array of weights.
            values (np.ndarray): Array of the values corresponding to each
                weight.
            capacity (int): Total capacity of the knapsack.
        """
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity

    @classmethod
    def from_csv(cls, filepath):
        """Load dataset from CSV file and return a KnapsackProblem object.

        Note: The CSV file is assumend to have three columns - weight, values
        and capacity, such as:

            Weights	             Values  	  Capacity
        [46 40 42 38 10]	[12 19 19 15  8]	40
        [11 31  4  6  7]	[ 2  8 18 16  3]	64

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            KnapsackProblem: Instance of the KnapsackProblem class with passed
                dataset as its fields.
        """
        df = pd.read_csv(filepath)

        # loading each parameters into numpy arrays
        weights = df["Weight"].values
        values = df["Value"].values
        capacity = df["Capacity"].iloc[0]

        return cls(weights, values, capacity)
