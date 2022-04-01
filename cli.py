import re
import inquirer
import numpy as np
from inquirer import errors


def validate_scalar(answers, current):
    """Validates if number is scalar"""
    if not re.fullmatch('-?\d+(\.\d+)?', current):
        raise inquirer.errors.ValidationError('', reason='This input has to be a scalar')
    return True


def validate_positive_int(answers, current):
    """Validates if number is an integer"""
    if not re.fullmatch('\d+', current):
        raise inquirer.errors.ValidationError('', reason='This input has to be a positive integer')
    return True


def validate_scalar_array(answers, current):
    """Validates a string for being an array of space separated scalars"""
    if not re.fullmatch('(-?\d+(\.\d+)?\s?)+', current):
        raise inquirer.errors.ValidationError('', reason='This input has to be single-space separated array')
    return True


def collect_choice(text, choices):
    return inquirer.list_input(text, choices=choices)


def collect_array(text, length):
    """Obtains an array of scalars from user"""
    print()
    array_str = inquirer.text(text, validate=validate_scalar_array).strip()
    array = [float(i) for i in array_str.split(' ')]
    if len(array) != length:
        print(f"Wrong value, make sure array has {length} elements. Try again")
        return collect_array(text, length)

    return np.array(array)


def collect_matrix(matrix_name, dim) -> np.matrix:
    print(f"Input matrix '{matrix_name}' row by row")
    print("each row should be a single-space separated array")
    rows = []
    for i in range(dim):
        rows.append(collect_array(f"Please input row #{i+1} (out of {dim})", dim))
    return np.matrix(rows)


def collect_number(text, force_int=False):
    """
    Obtains a number input from a user.
    If force_int flag is set, requires an integer input
    """
    validator = validate_scalar
    parser = float

    if force_int:
        validator = validate_positive_int
        parser = int

    print()
    num = inquirer.text(text, validate=validator)
    return parser(num)


def get_option(text, options: [(str, int)]) -> int:
    return inquirer.list_input(text, choices=options)


def is_positive_definite(matrix: np.array) -> bool:
    return np.all(np.linalg.eigvals(matrix) > 0)


def is_matrix_square(matrix):
    return matrix.shape[0] == matrix.shape[1]


def collect_matrix_A(dim) -> np.matrix:
    a = collect_matrix('A', dim)
    if not is_matrix_square(a):
        print('Error! The A matrix has to be square. Try Again.')
        return collect_matrix_A(dim)
    return a


def collect_vector_b(dim) -> np.array:
    return collect_array(f"Please input {dim}-element array b", dim)


def collect_scalar_c() -> float:
    return collect_number("Input the value of parameter 'c'")


def collect_dec_width() -> int:
    d = collect_number("Input the value of parameter 'd'", force_int=True)
    if d < 1:
        print('Error! value dimension has to be > 1. Try Again.')
        return collect_dec_width()

    return d


def collect_population_size() -> int:
    d = collect_number("Input the population size", force_int=True)
    if d < 2:
        print('Error! population size has to be at least 2. Try Again.')
        return collect_population_size()

    return d


def collect_crossover_probability() -> float:
    prob = collect_number("Input the crossover probability [0-1]")
    if not 0 <= prob <= 1:
        print('Error! value dimension has to be in [0-1] range. Try Again.')
        return collect_crossover_probability()
    return prob


def collect_mutation_probability() -> float:
    prob = collect_number("Input the mutation probability [0-1]")
    if not 0 <= prob <= 1:
        print('Error! value dimension has to be in [0-1] range. Try Again.')
        return collect_mutation_probability()
    return prob


def collect_matrix_dim() -> int:
    dim = collect_number("Input the dimensionality of the problem", force_int=True)
    if dim < 1:
        print('Error! Dimensionality has to be > 0. Try Again.')
        return collect_matrix_dim()
    return dim


def collect_iters_num() -> int:
    d = collect_number("Input iterations number", force_int=True)
    if d < 1:
        print('Error! Iterations number has to be > 1. Try Again.')
        return collect_iters_num()

    return d


def collect_params() -> dict:
    dim = collect_matrix_dim()
    A = collect_matrix_A(dim)
    b = collect_vector_b(dim)
    c = collect_scalar_c()

    pop = collect_population_size()
    cross_p = collect_crossover_probability()
    mutation_p = collect_mutation_probability()
    dec_width = collect_dec_width()

    iters_no = collect_iters_num()

    return {
        'A': A, 'b': b, 'c': c, 'dec_width': dec_width, 'pop': pop, 'dim': dim,
        'cross_p': cross_p, 'mutation_p': mutation_p, 'iters_no': iters_no
    }









