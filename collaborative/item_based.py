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
limit = 5  # A limit for the number of users to be common

neighbors = {}
averages = {}
deviations = {}


def get_books_ratings_deviation(b, users):
    rating_u = {user: user_book_rating[(user, b)] for user in users}
    average_u = np.mean(list(rating_u.values()))
    deviation_u = {user: r - average_u for user, r in rating_u.items()}
    deviation_u_values = np.array(list(deviation_u.values()))
    sigma_u = np.sqrt(deviation_u_values.dot(deviation_u_values))

    return (average_u, deviation_u, sigma_u)


for b in range(M):
    if b not in book_to_user:
        averages[b] = 0
        deviations[b] = []
        neighbors[b] = SortedList()
        continue
    b_users = set(book_to_user[b])
    average_u, deviation_u, sigma_u = get_books_ratings_deviation(b, b_users)

    averages[b] = average_u
    deviations[b] = deviation_u

    sl = SortedList()
    for c in range(M):
        if b != c:
            if c not in book_to_user:
                continue
            c_users = set(book_to_user[c])
            common_users = (b_users & c_users)
            if (len(common_users) > limit):
                average_v, deviation_v, sigma_v = get_books_ratings_deviation(
                    c, c_users)
                numerator = sum(deviation_u[b] * deviation_v[b]
                                for b in common_users)
                denominator = (sigma_u * sigma_v)
                if denominator:
                    w_bc = numerator / (sigma_u * sigma_v)

                    sl.add((-w_bc, c))
                    if (len(sl) > K):
                        del (sl[-1])
    neighbors[b] = sl
    if (b % 500 == 0):
        print("calculated neighbors for ", b)


def predict(b, u):
    numerator = 0
    denominator = 0
    for w_bc, j in neighbors[u]:
        try:
            numerator += -w_bc * deviations[j][u]
            denominator += abs(w_bc)
        except KeyError:
            pass

    if denominator == 0:
        prediction = averages[b]
    else:
        prediction = numerator / (denominator + averages[b])

    prediction = min(10, prediction)
    prediction = max(1, prediction)
    return prediction


def calculate(user_book_rating_map):
    predictions = []
    targets = []

    for (u, b), target in user_book_rating_map.items():
        prediction = predict(b, u)
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