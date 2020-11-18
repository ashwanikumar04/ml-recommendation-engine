from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from BXBook import BXBook

import random
import numpy as np

random_state = 10

np.random.seed(random_state)
random.seed(random_state)


def load_data():
    bx_book = BXBook()
    print("Loading book ratings...")
    data = bx_book.load_data()
    return (bx_book, data, {})


# Load up common data set for the recommender algorithms
(bx_book, evaluation_data, rankings) = load_data()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluation_data, rankings)

# Throw in an SVD recommender
SVDAlgorithm = SVD(random_state=10)
evaluator.add_algorithm(SVDAlgorithm, "SVD")

# Just make random recommendations
Random = NormalPredictor()
evaluator.add_algorithm(Random, "Random")

evaluator.evaluate()

evaluator.sample_top_n(bx_book)
