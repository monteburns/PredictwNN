import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


class Dataset():

    def __init__(self, name):
        if name == 'ua':
            self.df_ratings = pd.read_csv("ml-100k/ua.base", sep="\t", header=None,
                                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            self.df_users = pd.read_csv("ml-100k/u.user", sep='|', header=None,
                                        names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'])
            self.df = pd.merge(self.df_users, self.df_ratings, on='UserID')
        else:
            self.df_ratings = pd.read_csv("ml-100k/u.data", sep="\t", header=None,
                                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            self.df_users = pd.read_csv("ml-100k/u.user", sep='|', header=None,
                                        names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'])
            self.df = pd.merge(self.df_users, self.df_ratings, on='UserID')

    def all_numbers(self):
        self.df['Occupation'] = self.df['Occupation'].astype('category')

        encode_map = {
            'administrator': 1,
            'artist': 2,
            'doctor': 3,
            'educator': 4,
            'engineer': 5,
            'entertainment': 6,
            'executive': 7,
            'healthcare': 8,
            'homemaker': 9,
            'lawyer': 10,
            'librarian': 11,
            'marketing': 12,
            'none': 13,
            'other': 14,
            'programmer': 15,
            'retired': 16,
            'salesman': 17,
            'scientist': 18,
            'student': 19,
            'technician': 20,
            'writer': 21
        }
        self.df['Occupation'].replace(encode_map, inplace=True)

        self.df['Gender'] = self.df['Gender'].astype('category')
        encode_map = {
            'F': 1,
            'M': 0
        }

        self.df['Gender'].replace(encode_map, inplace=True)

        return True

    def drop_column(self, cname):
        self.df.drop(cname, inplace=True, axis=1)

        return True

    def cluster(self, base, n_cluster):
        if base == 'user':
            df_user = self.df.pivot(
                index='UserID',
                columns='MovieID',
                values='Rating'
            ).fillna(0)  # convert dataframe to scipy sparse matrix
            matrix = csr_matrix(df_user.values)
        elif base == 'movie':
            df_movie = self.df.pivot(
                index='MovieID',
                columns='UserID',
                values='Rating'
            ).fillna(0)  # convert dataframe to scipy sparse matrix
            matrix = csr_matrix(df_movie.values)
        else:
            print("Specify base as ['user'] or ['movie']")
            exit()

        kmeans = KMeans(
            init="random",
            n_clusters=n_cluster,
            n_init=10,
            max_iter=300,
            random_state=42
        )

        kmeans.fit(matrix)

        return kmeans.labels_

    def replace_all_users(self, labels):

        keys = [x + 1 for x in range(len(labels))]
        dest = dict(zip(keys, zip(labels)))

        self.df = self.df.replace({'UserID': dest})

        return True

    def replace_all_movies(self, labels):

        keys = [x + 1 for x in range(len(labels))]
        dest = dict(zip(keys, zip(labels)))

        self.df = self.df.replace({'MovieID': dest})

        return True

    def x_values(self):
        """ Returns userid, Gender, Age, Occupation, and MovieId """
        return self.df.iloc[:, 0:5].values

    def y_values(self):
        """Returns ratings"""
        return self.df.iloc[:, -1].values

    def values(self):
        """Returns values"""
        return self.df.values

    def test_values(self):
        df_ratings = pd.read_csv("ml-100k/ua.test", sep="\t", header=None,
                                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        df_users = pd.read_csv("ml-100k/u.user", sep='|', header=None,
                                    names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'])
        df_test = pd.merge(df_users, df_ratings, on='UserID')

        X_test = df_test.iloc[:, 0:5].values
        y_test = df_test.iloc[:, -1].values

        return X_test, y_test
