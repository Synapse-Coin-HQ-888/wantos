# Essential mathematical utilities for reactive visual systems
# Includes a parametric evaluator and a global computation environment for flexible mathematical execution

import random
import sys
import time
import typing
from math import *

import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, lfilter

# Global parameters
n = 1024  # Default array length
fps = 60  # Default frames per second
f = 0     # Global frame counter

math_env = {}
simplex = None
epsilon = 0.0001

SEED_MAX = 2**32 - 1

counters = dict(perlin01=0)

perlin_cache = []
max_perlin_cache = 10
enable_perlin_cache = False


def get_simplex():
    from opensimplex import OpenSimplex
    global simplex
    if simplex is None:
        simplex = OpenSimplex(random.randint(-999999, 9999999))
    return simplex


def prepare_math_env(*args):
    math_env.clear()
    for a in args:
        for k, v in a.items():
            math_env[k] = v


def update_math_env(k, v):
    math_env[k] = v
    globals()[k] = v


def reset():
    for k in counters.keys():
        counters[k] = 0
    perlin_cache.clear()


def set_seed(seed=None, with_torch=True):
    if seed is None:
        seed = int(time.time())
    elif isinstance(seed, str):
        seed = str_to_seed(seed)
    seed = seed % SEED_MAX
    seed = int(seed)
    global current_seed
    current_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        import torch
        torch.manual_seed(seed)
    pass


def str_to_seed(seed: str) -> int:
    import hashlib
    seed_hash = hashlib.sha1(seed.encode("utf-8"))
    seed_int = int(seed_hash.hexdigest(), 16)
    seed_int = seed_int % SEED_MAX
    return seed_int


def parametric_eval(string, **kwargs):
    if isinstance(string, (float, int)):
        return string
    elif isinstance(string, str):
        try:
            output = eval(string)
        except SyntaxError as e:
            raise RuntimeError(f"Error in parametric value: {string}")
        return output
    elif isinstance(string, list):
        return val_or_range(string)
    elif isinstance(string, tuple):
        return val_or_range(string)
    else:
        return string


def choose_or(l: list, default, p=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 0:
        return default
    else:
        ret = random.choices(l, p)
        if isinstance(ret, list):
            return ret[0]
        return ret


def choose(l: list, w=None, exclude=None):
    if exclude is not None:
        if w is None:
            w = [1] * len(l)
        w[exclude] = 0
    ret = random.choices(l, weights=w)
    if isinstance(ret, list):
        return ret[0]
    return ret


def choices(l: list, n=1, p=None):
    return random.choices(l, k=n, weights=p)


def rng(min_val: float | None = None, max_val: float | None = None):
    if min_val is None and max_val is None:
        return random.random()
    elif min_val is None and max_val is not None:
        return random.uniform(0, max_val)
    elif min_val is not None and max_val is None:
        return r
