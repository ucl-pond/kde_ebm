from ..mixture_model import get_prob_mat
from itertools import permutations
from ..event_order import EventOrder
import numpy as np


def enumerate_all(X, mixture_models):
    prob_mat = get_prob_mat(X, mixture_models)
    best_score = -1e10
    best_order = None
    enumerate_samples = []
    for sequence in permutations(np.arange(X.shape[1])):
        event_order = EventOrder(ordering=np.array(sequence))
        sequence_score = event_order.score_ordering(prob_mat)
        enumerate_samples.append(event_order)
        if sequence_score > best_score:
            best_score = sequence_score
            best_order = event_order
    return best_order
