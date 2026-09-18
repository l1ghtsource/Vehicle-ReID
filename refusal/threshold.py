import numpy as np

from .features import similarities


def max_cosine(query, gallery):
    sim = similarities(query, gallery)
    return sim.max(axis=1)


def gap12(query, gallery):
    sim = similarities(query, gallery)
    k = min(2, sim.shape[1])
    part = np.sort(sim, axis=1)
    top = part[:, -1]
    second = part[:, -k]
    return top - second
