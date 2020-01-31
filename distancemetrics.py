from constants import defaultkey
import numpy as np

def unit_dot_product(vec1, vec2):
    return np.dot(vec1, np.transpose(vec2))

options = {
    defaultkey: unit_dot_product,
    "dotprod": unit_dot_product
}

