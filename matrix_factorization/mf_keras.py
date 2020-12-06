import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# df = pd.read_csv("./data/processed_rating.csv")

# N = df["user_idx"].max() + 1
# M = df["isbn_idx"].max() + 1

# df = shuffle(df)

# cut_off = int(0.8 * len(df))

# df_train = df.iloc[:cut_off]
# df_test = df.iloc[cut_off:]

# K = 15

# mu = df_train["Book-Rating"].mean()
# epochs = 15
# reg_penalty = 0.0

# u = Input(shape=(1, ))
# b = Input(shape=(1, ))

# u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg_penalty))(u)
# b_embedding = Embedding(M, K, embeddings_regularizer=l2(reg_penalty))(b)

# u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg_penalty))(u)
# b_bias = Embedding(M, 1, embeddings_regularizer=l2(reg_penalty))(b)

# x = Dot(axes=2)([u_embedding, b_embedding])

# x = Add()([x, u_bias, b_bias])
# x = Flatten()(x)

# model = Model(inputs=[u, b], outputs=x)
# model.compile(loss='mse', optimizer=Adam(lr=0.01), metrics=["mse"])

# r = model.fit(
#     x=[df_train["user_idx"].values, df_train["isbn_idx"].values],
#     y=df_train["Book-Rating"].values - mu,
#     epochs=epochs,
#     batch_size=128,
#     validation_data=([df_test["user_idx"].values,
#                       df_test["isbn_idx"].values], df_test["Book-Rating"].values - mu))

# plt.plot(r.history['loss'], label="train loss")
# plt.plot(r.history['val_loss'], label="test loss")
# plt.legend()
# plt.show()

df = pd.read_csv("./data/archive/ratings.csv")

# N = len(set(df["user_id"].values)) + 1
# M = len(set(df["book_id"].values)) + 1

# df = shuffle(df)

# cut_off = int(0.8 * len(df))

# df_train = df.iloc[:cut_off]
# df_test = df.iloc[cut_off:]

# K = 15

# mu = df_train["rating"].mean()
# epochs = 15
# reg_penalty = 0.0

# u = Input(shape=(1, ))
# b = Input(shape=(1, ))

# u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg_penalty))(u)
# b_embedding = Embedding(M, K, embeddings_regularizer=l2(reg_penalty))(b)

# u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg_penalty))(u)
# b_bias = Embedding(M, 1, embeddings_regularizer=l2(reg_penalty))(b)

# x = Dot(axes=2)([u_embedding, b_embedding])

# x = Add()([x, u_bias, b_bias])
# x = Flatten()(x)

# model = Model(inputs=[u, b], outputs=x)
# model.compile(loss='mse', optimizer=Adam(lr=0.01), metrics=["mse"])

# r = model.fit(x=[df_train["user_id"].values, df_train["book_id"].values],
#               y=df_train["rating"].values - mu,
#               epochs=epochs,
#               batch_size=128,
#               validation_data=([
#                   df_test["user_id"].values, df_test["book_id"].values
#               ], df_test["rating"].values - mu))

# model.save('regression_model.h5')
# plt.plot(r.history['loss'], label="train loss")
# plt.plot(r.history['val_loss'], label="test loss")
# plt.legend()
# plt.show()


def predict(user_id):
    model =  keras.models.load_model('regression_model.h5')
    book_data = np.array(list(set(df.book_id)))
    user = np.array([user_id for i in range(len(book_data))])
    predictions = model.predict([user, book_data])
    predictions = np.array([a[0] for a in predictions])
    recommended_book_ids = (-predictions).argsort()[:5]
    print(recommended_book_ids)
    print(predictions[recommended_book_ids])

predict(1)
