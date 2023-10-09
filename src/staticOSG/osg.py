"""This script is an Python implementation of the Optimal Sequential Grouping algorithm described in this paper:
https://doi.org/10.1109/ISM.2016.0061
Additional functionalities were included to target the problem at hand: Segmenting a video to detect the frame blocks that belong to different rooms.
"""

import numpy as np
from typing import Union
np.random.seed(69)
import random
import logging
random.seed(69)
import matplotlib.pyplot as plt

## PART 1: creating the intermediate matrix
# first let's implement a helper matrix where the (i, j) position represents the sum of all pair-wise distances of elements (k, l) such that 
#  max(i, j) >= s >= min(i, j)  for s = k, l
def portion_sums_matrix(matrix: np.ndarray) -> np.ndarray:
    """ Given the initial matrix, this function builds a new matrix 
    where the element (i, j) is the sum of all values (k, l) where (max(i, j) >= s >= min(i, j)) for s = k, l

    Args:
        matrix (np.ndarray): a 2D array. This specific function does not make any assumptions. Nevertheless, further the code 
        assumes that the matrix represents the pair wise between given elements.

    Returns:
        np.ndarray: the result matrix  
    """

    # make sure the distances matrix is square 
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]

    # create the matrix
    sum_matrix = np.full(shape=matrix.shape, fill_value=None)
    N = matrix.shape[0]
    for i in range(N):
        sum_matrix[i][i] = matrix[i][i] # general expression

    # we will populate the matrix following the main diagonals
    for j in range(1, N):
        for i in range(0, N - j):
            sum_matrix[i][j + i] = matrix[i][j + i] + matrix[j + i][i] + sum_matrix[i + 1][j + i] + sum_matrix[i][j + i - 1] - (sum_matrix[i + 1][j + i - 1] if j > 1 else 0) 
            sum_matrix[j + i][i] = sum_matrix[i][j + i]
    
    # the initial type is object, converting the array to float can boost performance
    return sum_matrix.astype(float) 

## FIND THE BEST COMBINATION
def best_k_combination(K: int, distances: np.ndarray, N:int = None, return_optimal_cost:bool = False) -> Union[tuple[list[int], float], list[int]]:
    """Given 

    Args:
        K (int): The number of clusters / groups

        distances (np.ndarray): a 2D square array. The functionality of this code is not dependent on any assumptions. Nevertheless, the main use case assumes 
            that the (i, j) element of the 'distances' matrix represents a certain distance / dissimilarity measure 
            between the i-th and j-th element of a set of N elements  
        
        N (int, optional): the number of elements: must be equal the size of distances matrix. Defaults to None.
        return_optimal_cost (bool, optional): whether to return the optimal value of the cost function. Defaults to False.

    Returns:
        Union[tuple[list[int], float], list[int]]: the best combination and optionally the optimal value for the cost function
    """
    
    if N is None:
        N = distances.shape[0]
    
    assert len(distances.shape) == 2 and distances.shape[0] == distances.shape[1] and N == distances.shape[0] and K <= N

    # create the portion_sums matrix
    portion_sums = portion_sums_matrix(distances)

    # first we create the cost table
    cost_table = np.full(shape=(N, K + 1), fill_value=np.inf)
    
    # the position (i, j) of index_table saves the index of the first boudary of the optimal sequence 
    # if we start at i-th position and have only j clusters
    index_table = np.full(shape=(N, K + 1), fill_value=None)

    # set the base case: k = 0
    for i in range(N):
        # as there is only one cluster, the cost is simply the pair wise sum of all elements 
        cost_table[i][1] = portion_sums[i][N - 1]
        index_table[i][1] = N - 1

    for c in range(2, K + 1):
        for i in range(N - c + 1):
            # define the possible values 
            possible_costs = [portion_sums[i][l] + cost_table[l + 1][c - 1] for l in range(i, N - c + 1)] 
            # if possible_costs:
            best_index = np.argmin(possible_costs)
            cost_table[i][c] = possible_costs[best_index]
            # since the value best_index represents the index within possible_costs, we need to add i 
            index_table[i][c] = i + best_index 

    # to make sure the code works as expected
    optimal_cost = cost_table[0][K]
 
    # reconstructing the optimal sequence 
    boundaries = [index_table[0][K]]

    for i in range(2, K + 1):
        boundaries.append(index_table[boundaries[-1] + 1][K - i + 1])

    assert len(boundaries) == K and all([boundaries[i + 1] > boundaries[i] for i in range(len(boundaries) - 1)])

    # time to reconstruct the optimal cost
    constructed_cost = portion_sums[0][boundaries[0]]

    # constructed_cost = portion_sums[boundaries[0]][boundaries[1]]
    for i in range(len(boundaries) - 1):
        cost_add = portion_sums[boundaries[i] + 1][boundaries[i + 1]]
        constructed_cost += cost_add
    
    # passing this assert statement is a huge indicator of the code's correctness
    # assert optimal_cost == constructed_cost
    assert np.abs(optimal_cost - constructed_cost) < 1e-1

    if return_optimal_cost:
        return boundaries, optimal_cost

    return boundaries


def find_best_k(matrix: np.ndarray, debug: bool = False) -> int:
    """This is an implementation of the method suggested in the paper to find the optimal number of clusters for the OSG task. 
    Args:
        matrix (np.ndarray): a 2D square matrix. The functionality of the code does not make any assumptions. Nevertheless, the results
            are only meaningful if the matrix represents pair-wise distances / dissimilarities between different elements of a hidden set.
        debug (bool, optional): Whether to display a plot illustrating the code's inner mechanism. Defaults to False.

    Returns:
        int: the optimal number of clusters / groups.
    """

    # let's first apply the SVD procedure
    _, s, _ = np.linalg.svd(matrix, full_matrices=True)
    
    # let's extract the singular values that are larger than 1
    singular_values = [np.log(sv) for sv in s if sv > 1]
    m = len(singular_values)

    # if the value of m is equal to 1, then we should simply return '1'
    if m <= 1: return 1

    # let's define the main direction of the line
    direction_vector = np.array([[m - 1], [singular_values[-1] - singular_values[0]]])
    # define the start point vector
    start_vector = np.array([[1], [singular_values[0]]])
    
    if debug:
        # set the size of the figure
        plt.figure(figsize=(10, 10))
        # first plot the log values
        plt.plot(list(range(1, m + 1)), singular_values, label='singular values',)
        # plot the main direction
        X = [1, m]
        Y = [singular_values[0], singular_values[-1]]
        plt.plot(X, Y, label = 'main direction')
        
        def display_line(i: int):
            # first calculate the point vector
            point_vector = np.array([[i + 1], [singular_values[i]]])
            project_vector = direction_vector * ((direction_vector.T @ point_vector)/(direction_vector.T @ direction_vector)).squeeze()
            assert project_vector.shape == (2, 1)
            
            project_point_vector = start_vector + project_vector
            # get the indices to display
            Xs = [point_vector.squeeze()[0], project_point_vector.squeeze()[0]]
            Ys = [point_vector.squeeze()[1], project_point_vector.squeeze()[1]] 
            plt.plot(Xs, Ys, label = f'projection {i + 1}')

        # display each of the lines 
        for i, _ in enumerate(singular_values):
            display_line(i)
        plt.legend()
        plt.show()
    
    def point_distance(i: int):
        # define the vector of the current point
        point_vector = np.array([[i + 1], [singular_values[i]]])
        project_vector = direction_vector * ((direction_vector.T @ point_vector)/(direction_vector.T @ direction_vector)).squeeze()
        assert project_vector.shape == (2, 1)

        project_point_vector = start_vector + project_vector
        distance_vector = point_vector - project_point_vector

        assert distance_vector.shape == (2, 1)
        return np.linalg.norm(distance_vector)

    distances = [point_distance(i) for i, _ in enumerate(singular_values)]
    # calculate the different distances and return the index of the maximal one
    return np.argmax(distances) + 1 # as the list is 0-indexed


def build_distances_matrix(matrix: np.ndarray, horizontal: bool= True) -> np.ndarray:
    """Given a set of N elements saved in a 2D array, this function builds a matrix where
    the (i, j) element represents (1 - cosine similarity(i-th element, j-th element))

    Args:
        matrix (np.ndarray): The elements' matrix
        horizontal (bool, optional): True if the elements are represented by rows, false otherwise.. Defaults to True.
        
    Returns:
        np.ndarray: The distance / dissimilarity matrix
    """
    
    # only accept 2D arrays
    assert len(matrix.shape) == 2
    # if horizontal, we assume that each row represents an element
    if horizontal: 
        matrix = matrix.T
    # for computational efficiency, I make use of the fact that cosine similarity of 2 normalized vectors is their dot product.

    # now each element is represented by a column
    # the number of elements is the 2nd term in the shape
    n_elements = matrix.shape[1]
    norms = np.array([[np.linalg.norm(matrix[:, i]) for i in range(n_elements)]])
    
    # normalize
    distances = np.divide(matrix, norms)
    # make sure the norms of each column in the distances matrix is 1 (with negligeable error)
    assert all([np.abs(np.linalg.norm(distances[:, i]) - 1) <= 10 ** -3 for i in range(n_elements)])
    
    # extract the distances
    distances = 1 - distances.T @ distances
    return distances






def osg(matrix: np.ndarray, horizontal:bool=True, debug:bool = False, trials:int=4) -> list[int]:
    """This is an end to end interface of the Optimal Sequential Grouping algorithm.
    As the process of finding the best number of clusters relies on SVD which in turns bahaves differently depending on the matrix magnitude
    We multiply the distances matrix with different coefficients leading to different number of clusters.
    The final grouping is the one that minimizes the cost function computed on the original distances matrix (no coefficient multiplication) 

    Args:
        matrix (np.ndarray): The elements' matrix
        horizontal (bool, optional): True if the elements are represented by rows, false otherwise. Defaults to True.
        debug (bool, optional): whether to display a graphical representation of the find_best_k function. Defaults to False.
        trials (int, optional): The number of cofficients to experiment with. Defaults to 10.

    Returns:
        _type_: the list of indicies of the elements that represent the boundaries of each group
    """
        
    # first step is to build the distances matrix
    logging.getLogger("ServerApplication").info('Building the distances matrix...')
    distances = build_distances_matrix(matrix, horizontal=horizontal)
    # extract the max maig
    max_magnitude = int(np.amax(np.abs(matrix)))
    possible_cnts = np.linspace(1, max_magnitude, trials)
    assert len(possible_cnts) == trials
    
    best_cost, best_grouping = None, None 

    logging.getLogger("ServerApplication").info('Finding the best grouping...')
    for cte in np.linspace(1, max_magnitude, trials):
        # as norm of the matrix affects SVD, let's multiply by some constant
        # find the best_k
        logging.getLogger("ServerApplication").info('Finding the best K...')
        best_k = find_best_k(cte * distances, debug=debug)
        # find the optimal grouping
        logging.getLogger("ServerApplication").info('Finding the best K combination...')
        grouping_boundaries, cost = best_k_combination(best_k, distances, return_optimal_cost=True)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_grouping = grouping_boundaries
    
    return best_grouping








