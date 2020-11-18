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
    
    def load_data(self):

        # Look for files relative to the directory we are running from

        ratings_dataset = 0

        reader = Reader(line_format='user item rating', sep=';', skip_lines=1,  rating_scale= (0,10))

        ratings_dataset = Dataset.load_from_file(self.ratings_path, reader=reader)
        
        return ratings_dataset
    
    def get_book_name(self, book_id):
        return ""