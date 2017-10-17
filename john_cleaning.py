import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt


def clean_raw_data(filename):
    """
    Takes the joke file, picks out the jokes, removes the html and space characters from them
    INPUT:
    filename(string) - file path to joke file
    OUTPUT:
    list - list of jokes (as strings) in original order
    """
    with open(filename, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
    messy = [x.text for x in soup.select('p')]
    cleaned = [x.replace('\n', ' ').replace('\r', ' ') for x in messy]
    return cleaned

def convert_to_tdidf(lst):
    """
    Converts a list of strings to a tdidf feature matrix and feature name map
    INPUT:
    lst(list) - list of sentences to be turned into the tdidf vectors
    OUTPUT:
    vectors(numpy array) - sparse matrix of vectors for each sentence (rows:sentence, column: word tfidf)
    feature_names(list) - list of feature names used in vectors
    """
    snowball = SnowballStemmer('english')
    stemmed = [snowball.stem(item) for item in lst]
    tfidf = TfidfVectorizer(stop_words='english', max_features=None)
    vectors = tfidf.fit_transform(stemmed)
    vectors = vectors.toarray()
    feature_names = tfidf.get_feature_names()
    return vectors, feature_names

def matrix_shrinkage(matrix, power_level=0.9):
    """
    INPUTS:
    matrix(numpy array) - the matrix to be reduced
    power_level(float) - the percentage of explained variance desired (b/t 0 and 1)
    OUTPUTS:
    U_trunc - the trunkated user weight array (rows: items, columns: latent features, values: weights)
    Sigma_trunc - the trunkated power array (rows: latent features, columns: power, values: power)
    VT_trunc - the truncated features array (rows: item features, columns: latent features, values: weights)
    """
    if power_level>1:
        print "No. Power level can't be more than 1. Setting power_level to 0.9"
        power_level=0.9
    #decompose matrix into U,S,V components
    U,Sigma,VT = np.linalg.svd(matrix)
    #shrink matrix to latent features that account for power_level fraction of total power
    power = Sigma**2
    passed_thresh = np.argwhere(np.cumsum(power)/np.sum(power) >= power_level)
    U_trunc = U[:, :passed_thresh[0][0]]
    Sigma_trunc = Sigma[:passed_thresh[0][0]]
    VT_trunc = VT[:, :passed_thresh[0][0]]
    return U_trunc, Sigma_trunc, VT_trunc


if __name__ == '__main__':

    filename = 'data/jokes.dat'

    list_of_jokes = clean_raw_data(filename)

    joke_tdidf_vectors, feature_names = convert_to_tdidf(list_of_jokes)


    joke_tdidf_shrunk, latent_feature_powers, word_impact = matrix_shrinkage(joke_tdidf_vectors, power_level=0.9)
