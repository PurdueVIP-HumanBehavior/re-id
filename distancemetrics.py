import numpy as np

def unitdotprod(vec1, vec2):
    return np.dot(vec1, np.transpose(vec2))

options = {
    "default": unitdotprod,
    "dotprod": unitdotprod
}

