import itertools

from surprise import accuracy
from collections import defaultdict


class RecommenderMetrics:

    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)

    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)
