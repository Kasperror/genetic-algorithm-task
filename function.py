import numpy as np


def func_val(x, A: np.ndarray, b: np.ndarray, c: float) -> float:
    return x.T.dot(A).dot(x) + b.T.dot(x) + c


def fitness_func(x: np.array, A: np.matrix, b: np.ndarray) -> float:
    assert len(b) == len(x) == A.shape[0]
    # c is omitted due to no real impact of the solution
    # and for huge numbers could slow down the computations
    return (x.T.dot(A).dot(x) + b.T.dot(x)).item()
