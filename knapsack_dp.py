from knapsack.dataset import DataInterface

# create dataset object from csv file
dataset = DataInterface.from_csv("datasets/dataset.csv")


def knapsack_dp(weights, values, capacity):
    """
    Solves the 0/1 knapsack problem using dynamic programming.

    Parameters:
    - weights (list): List of weights of each item.
    - values (list): List of values of each item.
    - capacity (int): Maximum weight capacity of the knapsack.
    Returns
    - int: Maximum value that can be achieved without exceeding the capacity.
    """
    # gene length
    n = len(weights)

    # dp table capacity x gene length
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # fill in the dp table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # if weight is less than capacity, it can be included
                dp[i][w] = max(
                    dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]]
                )
            else:
                # if greater than, cannot be included
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity], dp


# Define the problem parameters
weights = [2, 3, 4, 5, 9]
values = [3, 4, 8, 8, 10]
capacity = 10

# Call the knapsack function
max_value, dp_table = knapsack_dp(weights, values, capacity)

__import__("pprint").pprint(dp_table)
