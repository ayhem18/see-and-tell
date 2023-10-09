""" THIS SCRIPT IS USED TO TEST THE Optimal Sequence Grouping implementation written in the osg.py script.
The testing procedure consists of:
1. writing a naive, brute-force version of each functionality
2. compare the results of the optimized code with the brute force implementation 
3. leave the testing code running for several hours.
"""

import numpy as np
from osg import portion_sums_matrix, best_k_combination
from itertools import combinations
np.random.seed(69)
import random
random.seed(69)

# PART 1: TESTING THE PORTION SUM MATRIX (explained in the osg.py script)

def portion_sums_matrix_naif(matrix: np.ndarray) -> np.ndarray:
    # make sure the distances matrix is square 
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]

    # create the return matrix
    sum_matrix = np.full(shape=matrix.shape, fill_value=None)
    N = matrix.shape[0]

    for i in range(N):
        for j in range(N):
            sum_matrix[i][j] = np.sum(matrix[min(i, j): max(i,j) + 1, min(i, j):max(i, j) + 1])
    return sum_matrix

def test_sum_matrix():
    for i in range(1000):
        # get  a random shape
        N = np.random.randint(5, 250)
        random_matrix = np.random.randint(-5, 100, size=(N, N))
        # calculate the matrix in naif way
        true_matrix = portion_sums_matrix_naif(random_matrix)
        calculated_matrix = portion_sums_matrix(random_matrix)

        assert np.array_equal(true_matrix, calculated_matrix)


# PART2 : TESTING THE GROUPING ALGORITHM IMPLEMENTATION


def best_k_combination_naive(K: int, distances: np.ndarray, N:int = None, return_optimal_cost:bool = False):
    """this functions finds the combination of K groups that minimizes the    

    Args:
        K (int): the number of clusters / groups 
        distances (np.ndarray): a matrix where element (i, j) represents the distance between the i-th and j-th elements
        N (int, optional): The size of the distances matrix. Defaults to None.
        return_optimal_cost (bool, optional): whether to return the optimal value of the cost function or not. Defaults to False.
    """

    if N is None:
        N = distances.shape[0]
    
    assert len(distances.shape) == 2 and distances.shape[0] == distances.shape[1] and N == distances.shape[0] and K <= N

    # create the portion_sums matrix
    portion_sums = portion_sums_matrix(distances)

    # find all the possible combinations of k - 1 out of the indices
    all_combs = combinations(list(range(0, N - 1)), K - 1) if K > 1 else [[]]

    best_score = None
    best_comb = None

    for com in all_combs:
        # calculate the score for the combination
        comb_score = portion_sums[0, com[0]] if com else 0
        comb_score += sum([portion_sums[com[i] + 1][com[i + 1]] for i in range(len(com) - 1)])
        comb_score += portion_sums[(com[-1] + 1 if com else 1), N - 1] 
    
        if best_score is None or comb_score < best_score:
            best_score = comb_score
            best_comb = list(com)
    
    # don't forget to add the last boundary: last index in the list
    best_comb.append(N - 1)

    if return_optimal_cost:
        return best_comb, best_score
    
    return best_comb


def test_best_combination(N: int):
    # as the time complexity of the brute force implementation can be approximated to n ^ k / k!
    # the value of n should not be too large.
    matrix = np.random.randint(0, 50, size=(N, N))
    for k in range(2, min(7, N + 1)):
        comb1, c1 = best_k_combination_naive(k, matrix, return_optimal_cost=True)
        comb2, c2 = best_k_combination(k, matrix, return_optimal_cost=True)
        assert  c1 == c2 and comb1 == comb2
        print(f"CODE PASSSED TEST FOR k: {k}")
        print()


