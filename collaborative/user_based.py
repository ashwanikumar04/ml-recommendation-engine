import utils.IOUtils as io
import pandas as pd
import numpy as np
from sortedcontainers import SortedList

DATA_PATH = "./data/"

user_to_book = io.load_obj(DATA_PATH + "user_to_book")
book_to_user = io.load_obj(DATA_PATH + "book_to_user")
user_book_rating = io.load_obj(DATA_PATH + "user_book_rating")
user_book_rating_test = io.load_obj(DATA_PATH + "user_book_rating_test")

n1 = np.max(list(user_to_book.keys())) + 1
n2 = np.max([u for (u, b), r in user_book_rating.items()])
N = max(n1, n2) + 1

m1 = np.max(list(book_to_user.keys())) + 1
m2 = np.max([b for (u, b), r in user_book_rating.items()])

M = max(m1, m2) + 1

print("Users: ", N, "Books: ", M)

K = 25  #number of neighbors to consider
limit = 5  # A limit for the number of books to be common

print([u for (u, b), r in user_book_rating.items() if u not in user_to_book])

print(
    sorted(
        set([
            u for (u, b), r in user_book_rating_test.items()
            if u not in user_to_book
        ])))

neighbors = {}
averages = {}
deviations = {}


def get_user_ratings_deviation(u, books):
    rating_u = {book: user_book_rating[(u, book)] for book in books}
    average_u = np.mean(list(rating_u.values()))
    deviation_u = {book: r - average_u for book, r in rating_u.items()}
    deviation_u_values = np.array(list(deviation_u.values()))
    sigma_u = np.sqrt(deviation_u_values.dot(deviation_u_values))

    return (average_u, deviation_u, sigma_u)


for u in range(N):
    if u not in user_to_book:
        averages[u] = 0
        deviations[u] = []
        neighbors[u] = SortedList()
        continue
    u_books = set(user_to_book[u])
    average_u, deviation_u, sigma_u = get_user_ratings_deviation(u, u_books)

    averages[u] = average_u
    deviations[u] = deviation_u

    sl = SortedList()
    for v in range(N):
        if u != v:
            if v not in user_to_book:
                continue
            v_books = set(user_to_book[v])
            common_books = (u_books & v_books)
            if (len(common_books) > limit):
                average_v, deviation_v, sigma_v = get_user_ratings_deviation(
                    v, v_books)
                numerator = sum(deviation_u[b] * deviation_v[b]
                                for b in common_books)
                denominator = (sigma_u * sigma_v)
                if denominator:
                    w_uv = numerator / (sigma_u * sigma_v)

                    sl.add((-w_uv, v))
                    if (len(sl) > K):
                        del (sl[-1])
    neighbors[u] = sl
    if (u % 500 == 0):
        print("calculated neighbors for ", u)


def predict(u, b):
    numerator = 0
    denominator = 0
    for w_uv, j in neighbors[u]:
        try:
            numerator += -w_uv * deviations[j][b]
            denominator += abs(w_uv)
        except KeyError:
            pass

    if denominator == 0:
        prediction = averages[u]
    else:
        prediction = numerator / (denominator + averages[u])

    prediction = min(10, prediction)
    prediction = max(1, prediction)
    return prediction


def calculate(user_book_rating_map):
    predictions = []
    targets = []

    for (u, b), target in user_book_rating_map.items():
        prediction = predict(u, b)
        predictions.append(prediction)
        targets.append(target)

    return predictions, targets


def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)


train_predictions, train_targets = calculate(user_book_rating)
print('train mse:', mse(train_predictions, train_targets))

test_predictions, test_targets = calculate(user_book_rating_test)
print('test mse:', mse(test_predictions, test_targets))