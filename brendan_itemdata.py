import graphlab
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from john_cleaning import *

#Running John's python file to return TFIDF Vector Matrix
filename = 'data/jokes.dat'
list_of_jokes = clean_raw_data(filename)
vectors, features = convert_to_tdidf(list_of_jokes)
tfidf_matrix = vectors

#reducing tfidf matrix
u, sigma, vt = matrix_shrinkage(tfidf_matrix, power_level=0.3)

#Converting TFIDF Matrix to a Pandas DataFrame
itemdata_df = pd.DataFrame(u)
itemdata_df.index += 1
itemdata_df['joke_id'] = itemdata_df.index


#Graphlab Recommender Instantiated
ratings_contents = pd.read_table("data/ratings.dat")
ratings_contents['rating'] = ratings_contents['rating'] + 10
ratings_contents['rating'] = ratings_contents['rating']**2

sf = graphlab.SFrame(ratings_contents)
itemdata_sf = graphlab.SFrame(itemdata_df)
m1 = graphlab.recommender.item_similarity_recommender.create(sf, user_id = 'user_id', item_id = 'joke_id',
                                                           target = 'rating', item_data=itemdata_sf, similarity_type='pearson')

 #predict one:
one_datapoint_itemdata_sf = graphlab.SFrame({'user_id': [49541, 39499], 'joke_id': [113, 37]})
m1.predict(one_datapoint_itemdata_sf)

#Read in test set:
test_set = pd.read_csv('data/sample_submission.csv')
testusers = test_set['user_id'].tolist()
testjokes = test_set['joke_id'].tolist()

#predictions:
test_predict_sf = graphlab.SFrame({'user_id': testusers, 'joke_id': testjokes})
#m1.predict(test_predict_sf)
predictions = m1.predict(test_predict_sf)

test_output_df = test_set.copy()
test_output_df['rating'] = list(predictions)
test_output_df.to_csv('output8.csv')
