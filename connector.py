import numpy as np
from math import ceil
from function import fitness_func
from genetics import DEFAULT_CHROM_SIZE, Chromosome

"""
Representing floats in binary as a concatenation 
of binary array representations of integer part and floating point part
"""


def calculate_chrom_size(dec_width: int, fp_w: int, dim: int):
    return (1 + dec_width + fp_w) * dim


# for testing only
def float_to_b_array(num: float, dec_width, fp=0) -> list[int]:
    """Converts a chromosome into the corresponding x-vector value"""

    dec = int(num)
    unsigned_dec_in_bin = bin(dec).replace("0b", "")

    sign = "0"
    if unsigned_dec_in_bin[0] == "-":
        sign = "1"
        unsigned_dec_in_bin = unsigned_dec_in_bin[1:]

    assert len(unsigned_dec_in_bin) <= dec_width

    bin_str = sign + unsigned_dec_in_bin.zfill(dec_width)

    assert isinstance(fp, int)
    assert fp >= 0

    if fp > 0:
        # adding a floating point value to the Chromosome
        floating = int((num-dec)*10**fp)
        fp_in_bin = bin(floating).replace("0b", "")
        # ceil(3.322*fp) is approximation brought from log2(10^x)
        # which always gives ~3.322
        bin_str += fp_in_bin.zfill(ceil(3.322 * fp))

    # sign|unsigned binary form of integer part|binary form of floating point
    return [int(val) for val in bin_str]


def b_array_to_float(b_array: list, dec_width):
    """
    Converts the binary array generated using 'float_to_bin'
    back to flot value
    """
    assert len(b_array) > dec_width

    # casting to string for easier int casting
    b_array = ''.join([str(val) for val in b_array])

    sign = int(b_array[0])
    # removing sign character for easier, non-padded array access
    b_array = b_array[1:]
    dec = int(b_array[:dec_width], 2)

    num = dec
    # setting appropriate sign
    if sign:
        num *= -1

    if len(b_array) > dec_width:
        # adding the floating part to decimal part
        floating_as_int = int(b_array[dec_width:], 2)
        floating = (floating_as_int/(10**len(str(floating_as_int))))
        num += floating

    return num


def chrom_to_x(chrom: Chromosome, dim: int, dec_width) -> np.array:
    """Converts a chromosome into the corresponding x-vector value"""
    assert dim >= 1
    genes = chrom.genes
    # assuming the genes logically consists of dim*dim equal elements
    # in a form of binary vectors

    split_every = int(len(genes)/dim)  # values end every vals_no
    # print(f'{split_every=}')

    # splitting array into n equal elements where n is dim*dim
    bin_vals_list = [genes[i:i+split_every] for i in range(0, len(genes), split_every)]
    # print(f'{bin_vals_list=}')
    # calculating values for each element
    vals_list = [b_array_to_float(bin_val, dec_width=dec_width) for bin_val in bin_vals_list]
    # x = np.matrix([vals_list[i:i+dim] for i in range(0, len(vals_list), dim)])
    x = np.array(vals_list)
    return x


def chrom_fitness_func(chrom: Chromosome, A: np.matrix, b: np.ndarray, dec_width: int) -> float:
    """Returns fitness value of a chromosome"""
    dim = A.shape[0]
    x = chrom_to_x(chrom, dim=dim, dec_width=dec_width)
    return fitness_func(x=x, A=A, b=b)



if __name__ == "__main__":
    # b_array = []
    # for i in range(9):
    #     b_array.extend(float_to_b_array(2.244, dec_width=10, fp=3))

    test_num = 2.224
    test_width = 10
    test_fp = 2

    genes = float_to_b_array(test_num, dec_width=test_width, fp=test_fp)
    chrom = Chromosome(genes=genes)
    print(f'{chrom.genes=}')
    float_again = b_array_to_float(chrom.genes, dec_width=test_width)
    print(f'{float_again=}')

    test_A = np.matrix([-1])
    test_b = np.array([1])

    test_genes = float_to_b_array(2, test_width, test_fp)
    test_chrom = Chromosome(genes=test_genes)
    fit = chrom_fitness_func(test_chrom, A=test_A, b=test_b, dec_width=test_width)
    print(f'{fit=}')
    print(chrom_to_x(Chromosome(genes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]), 1, 10))





























