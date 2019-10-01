import numpy as np

def dotprod(vec1, vec2):
    return np.dot(vec1, np.transpose(vec2))

options = {
    "default": dotprod,
    "dotprod": dotprod
}