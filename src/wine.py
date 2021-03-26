import os
import pandas as pd

script_dir = os.path.dirname(__file__)
data_dir = script_dir.replace('src', '') + 'data/'
wine_file_name = 'winemag-data-130k-v2.csv'
wine_file = data_dir + wine_file_name 

reviews = pd.read_csv(wine_file, index_col=0)

print(reviews.head)