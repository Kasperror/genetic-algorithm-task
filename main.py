import itertools
from itertools import count


iter_no_limit = 1000
assert iter_no_limit >= 1

# dimensionality of x
d = 3

# c can be omitted as crossover probability is normalized
chromosome_size = d * d + d


for i in itertools.count():

    # stopping condition
    if i >= iter_no_limit:
        break








