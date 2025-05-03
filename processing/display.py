import os
from processing import preprocess
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



def cosine_similarity_manual(A, B=None):
    if B is None:
        B = A
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    norm_product = A_norm @ B_norm.T
    sim = (A @ B.T) / (norm_product + 1e-10)
    return sim

class CustomKNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None

    def fit(self, X):
        self.X = X

    def kneighbors(self, query_vec):
        sims = cosine_similarity_manual(query_vec, self.X)
        top_k_indices = np.argsort(-sims, axis=1)[:, 1:self.k+1]
        return top_k_indices

class Main():

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __init__(self):
        self.new_df = None
        self.movies = None
        self.movies2 = None
        self.knn_models = {}
        self.knn_neighbors = {}
        self.cv = None
        self.vec_store = {}  # Cache vectorized data


    def getter(self):
        return self.new_df, self.movies, self.movies2

    def get_df(self):
        pickle_file_path = r'Files/new_df_dict.pkl'

        # Checking if preprocessed dataframe already exists or not
        if os.path.exists(pickle_file_path):

            # Read the Pickle file and load the dictionary -- 3 times
            # For the movies dataframe
            pickle_file_path = r'Files/movies_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)

            self.movies = pd.DataFrame.from_dict(loaded_dict)

            # Now, for the movies2 doing the same work
            pickle_file_path = r'Files/movies2_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict_2 = pickle.load(pickle_file)

            self.movies2 = pd.DataFrame.from_dict(loaded_dict_2)

            # Now, For new_df
            pickle_file_path = r'Files/new_df_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)

            self.new_df = pd.DataFrame.from_dict(loaded_dict)

        else:
            self.movies, self.new_df, self.movies2 = preprocess.read_csv_to_df()

            # Converting to pickle file (dumping file)
            # Convert the DataFrame to a dictionary

            #  Now, doing for the movies dataframw
            movies_dict = self.movies.to_dict()

            pickle_file_path = r'Files/movies_dict.pkl'
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(movies_dict, pickle_file)

            #  Now, doing for the movies2 dataframe
            movies2_dict = self.movies2.to_dict()

            pickle_file_path = r'Files/movies2_dict.pkl'
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(movies2_dict, pickle_file)

            # For the new_df
            df_dict = self.new_df.to_dict()

            # Save the dictionary to a Pickle file
            pickle_file_path = r'Files/new_df_dict.pkl'
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(df_dict, pickle_file)
        ...

    def vectorise(self, col_name):
        if col_name not in self.vec_store:
            self.cv = CountVectorizer(max_features=5000, stop_words='english')
            self.vec_store[col_name] = self.cv.fit_transform(self.new_df[col_name]).toarray()
        return self.vec_store[col_name]


    def get_similarity(self, col_name):
        pickle_file_path = fr'Files/similarity_tags_{col_name}.pkl'
        if not os.path.exists(pickle_file_path):
            vec_tags = self.vectorise(col_name)
            similarity_tags = cosine_similarity_manual(vec_tags)

            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(similarity_tags, pickle_file)


    # ✅ NEW: Fit a KNN model based on feature (like 'tags', 'genres')
    def get_knn_model(self, col_name, k=10):
        vec_tags = self.vectorise(col_name)
        model = CustomKNN(k=k)
        model.fit(vec_tags)
        self.knn_models[col_name] = model
        self.knn_neighbors[col_name] = model.kneighbors(vec_tags)

    # ✅ NEW: Recommend using KNN
    def get_knn_recommendations(self, movie_title, col_name='tags', k=5):
        if col_name not in self.knn_models:
            self.get_knn_model(col_name, k=k)

        movie_idx = self.new_df[self.new_df['title'] == movie_title].index[0]
        vec_tags = self.vectorise(col_name)
        query_vec = vec_tags[movie_idx].reshape(1, -1)

        indices = self.knn_models[col_name].kneighbors(query_vec)[0]
        recommended_titles = self.new_df.iloc[indices]['title'].tolist()
        return recommended_titles


    def main_(self):
        self.get_df()
        self.get_similarity('tags')
        self.get_similarity('genres')
        self.get_similarity('keywords')
        self.get_similarity('tcast')
        self.get_similarity('tprduction_comp')
