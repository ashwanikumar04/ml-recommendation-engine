from surprise import AlgoBase
from surprise import PredictionImpossible
from base.BXBook import BXBook
import math
import numpy as np
import heapq


class ContentBasedKNN(AlgoBase):
    def __init__(self, k=30):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        bx_book = BXBook()
        genres = bx_book.getGenres()
        years = ml.getYears()