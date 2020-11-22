import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np


class BXBook:

    ratings_path = './data/BX-Book-Ratings-10K.csv'
    books_path = './data/BX-Books.csv'

    def load_data(self):

        # Look for files relative to the directory we are running from
        self.book_id_to_name_mapping = {}

        ratings_dataset = 0

        reader = Reader(line_format='user item rating',
                        sep=';',
                        skip_lines=1,
                        rating_scale=(0, 10))

        ratings_dataset = Dataset.load_from_file(self.ratings_path,
                                                 reader=reader)
        with open(self.books_path, newline='',
                  encoding='ISO-8859-1') as csvfile:
            book_reader = csv.reader(csvfile, delimiter=';')
            next(book_reader)  #Skip header line
            for row in book_reader:
                book_id = row[0]
                book_name = row[1]
                self.book_id_to_name_mapping[book_id] = book_name
        return ratings_dataset

    def get_book_name(self, book_id):
        return self.book_id_to_name_mapping.get(book_id, "")
