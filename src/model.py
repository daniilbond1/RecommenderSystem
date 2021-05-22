import numpy as np
import pandas as pd
import surprise
import sklearn 

from sklearn.preprocessing import MinMaxScaler
    
class PopularityRecommender():
    
    def fit(self, X, y):
        self.X=X
        Ratings=X
        Ratings['rating']=y
        self.movie_ids = Ratings['movieId'].unique()
        Ratings=Ratings.set_index('movieId')
        summ=[]
        for i in range(len(self.movie_ids)):
            summ.append(Ratings.loc[self.movie_ids[i]]['rating'].sum())
        film_weights=pd.DataFrame({'movieId': self.movie_ids, 'weight': summ})
        self.weights=film_weights
        scaler = MinMaxScaler()
        self.weights['weight']=scaler.fit_transform(self.weights['weight'].to_numpy().reshape(-1, 1))
        self.weights=dict(zip(self.weights['movieId'],self.weights['weight']))

    def predict_one(self, user_id, item_id):
        if item_id in self.weights:
            return self.weights[item_id]
        else:
            return np.mean(list(self.weights.values()))
        
        
    def predict(self, X):
        y = []
        for i,row in X.iterrows():
            y.append(self.predict_one(row['userId'], row['movieId']))
        return y
        
    def recommend_items(self, user_id, n_items=10):
        movies=self.X
        movies=movies.drop_duplicates(subset=['movieId'])
        movies = movies.drop(columns=["userId"])
        Ratings=self.X
        watched=list(Ratings[Ratings['userId']==user_id]['movieId'])
        movie_ids = np.array(self.movie_ids)
        non_watched=movie_ids[np.where(~np.isin(movie_ids, watched))]
        df_non_watched=pd.DataFrame({'movieId': non_watched, 'userId': [user_id]*len(non_watched)})
        df_non_watched  = df_non_watched.merge(movies, left_on='movieId', right_on='movieId')
        print(df_non_watched.columns)
        y=self.predict(df_non_watched)
        df_non_watched["score"] = y
        df_non_watched.sort_values(by="score", ascending=False, inplace=True)
        print(y)
        return list(df_non_watched.iloc[:n_items]["title"])
    
    def get_most_popular(self, n_items=10):
        movies=self.X
        movies=movies.drop_duplicates(subset=['movieId'])
        pop=pd.DataFrame.from_dict({'movieId': self.weights.keys(), 'weight': self.weights.values()})
        pop=pop.merge(movies, left_on='movieId', right_on='movieId',)
        return pop.sort_values(by='weight', ascending=False).head(n_items)



from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

class ContentRecommender():
    
    def fit(self, X, y):
        self.X=X
        Ratings=X
        movies=X
        Ratings['rating']=y
        self.movie_ids = Ratings['movieId'].unique()
        self.tfidf = TfidfVectorizer(stop_words='english')
        movies=movies.drop_duplicates(subset=['movieId'])
        overview_matrix = self.tfidf.fit_transform(movies['title']).toarray()
        self.film_vectors=dict()
        for i in range(len(self.movie_ids)):
            self.film_vectors[self.movie_ids[i]]=overview_matrix[i]
        
        user_ids=Ratings['userId'].unique()
        self.user_vectors=dict()
        for ID in user_ids:
            user_movies=Ratings[Ratings['userId'] == ID]['movieId'].to_numpy()
            user_raitings=Ratings[Ratings['userId'] == ID]['rating'].to_numpy()
            summ=0
            for i in range(len(user_raitings)):
                summ+=user_raitings[i]*self.film_vectors[user_movies[i]]
            self.user_vectors[ID]=summ

    def predict_one(self, user_id, item_id):
        if item_id in self.film_vectors:
            return cosine(self.user_vectors[user_id],self.film_vectors[item_id])
        else:
            return 0
        
    def predict(self, X):
        y = []
        for i,row in X.iterrows():
            y.append(self.predict_one(row['userId'], row['movieId']))
        return y
        
    def recommend_items(self, user_id, n_items=10):
        movies=self.X
        movies=movies.drop_duplicates(subset=['movieId'])
        movies = movies.drop(columns=["userId"])
        Ratings=self.X
        watched=list(Ratings[Ratings['userId']==user_id]['movieId'])
        movie_ids = np.array(self.movie_ids)
        non_watched=movie_ids[np.where(~np.isin(movie_ids, watched))]
        df_non_watched=pd.DataFrame({'movieId': non_watched
                                     , 'userId': [user_id]*len(non_watched)})
        df_non_watched  = df_non_watched.merge(movies)
        y=self.predict(df_non_watched)
        df_non_watched["score"] = y
        df_non_watched.sort_values(by="score", ascending=True, inplace=True)
        return list(df_non_watched.iloc[:n_items]["title"])
    
    def features(self,user_id, n_features=10):
        voc=dict(sorted(self.tfidf.vocabulary_.items(), key=lambda x: x[1]))
        h=pd.DataFrame({'user_vector':self.user_vectors[user_id],'features':voc.keys()})
        return h.sort_values(by='user_vector',ascending=False)



from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV

class CollaborativeFilteringRecommender():
    
    def fit(self, X, y):
        self.X=X
        Ratings=X
        movies=X
        Ratings['rating']=y
        self.movie_ids = Ratings['movieId'].unique()
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(Ratings[['userId','movieId','rating']], reader)
        param_grid = {'n_factors':[50,100],'n_epochs':[30],  'lr_all':[0.005],'reg_all':[0.02]}
        self.gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'],refit=True,cv=3)
        self.gs.fit(data)


    def predict_one(self, user_id, item_id):
        return self.gs.predict(user_id,item_id).est
        
    def predict(self, X):
        y = []
        for i,row in X.iterrows():
            y.append(self.predict_one(row['userId'], row['movieId']))
        return y
        
    def recommend_items(self, user_id, n_items=10):
        movies=self.X
        movies=movies.drop_duplicates(subset=['movieId'])
        movies = movies.drop(columns=["userId"])
        Ratings=self.X
        watched=list(Ratings[Ratings['userId']==user_id]['movieId'])
        movie_ids = np.array(self.movie_ids)
        non_watched=movie_ids[np.where(~np.isin(movie_ids, watched))]
        df_non_watched=pd.DataFrame({'movieId': non_watched
                                     , 'userId': [user_id]*len(non_watched)})
        df_non_watched  = df_non_watched.merge(movies)
        y=self.predict(df_non_watched)
        df_non_watched["score"] = y
        df_non_watched.sort_values(by="score", ascending=False, inplace=True)
        return list(df_non_watched.iloc[:n_items]["title"])
