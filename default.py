import numpy as np

# A = np.matrix([[-1, 0],
#                [0, -1]])
#
# b = np.array([1, 1])

# -x^2+2x -> x_opt = 1 val_opt = 1
A = np.matrix([-1])
b = np.array([2])


assert A.shape[0] == A.shape[1] == len(b)

DEFAULT_PARAMS = {
  'A': A,
  'b': b,
  'c': 0,
  'dim': A.shape[0],
  'dec_width': 3,
  'pop': 50,
  'cross_p': 0.8,
  'mutation_p': 0.05,
  'iters_no': 300
}