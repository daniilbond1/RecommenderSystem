from flask import Blueprint, request
import pandas as pd

from src.model import PopularityRecommender
from src.model import ContentRecommender
from src.model import CollaborativeFilteringRecommender

pr = PopularityRecommender()
cr = ContentRecommender()
cfr = CollaborativeFilteringRecommender()

def build_df(movies,ratings):
    movies=movies.drop('genres', axis=1)
    ratings=ratings.drop('timestamp', axis=1)
    df=ratings.merge(movies, left_on='movieId', right_on='movieId',)
    y=df['rating']
    X=df.drop('rating', axis=1)
    return X,y

movies = pd.read_csv('./data/movies.csv')
ratings = pd.read_csv('./data/ratings.csv')

X, y = build_df(movies,ratings)
pr.fit(X, y)
cr.fit(X, y)
cfr.fit(X, y)

controller = Blueprint(name="controller",
                       import_name=__name__,
                       url_prefix="/recommend")

from flask import jsonify
def response_success(body):
    return jsonify({"body": str(body)}), 200

@controller.route(rule="/", methods=["POST"])
def recommend():

    method = request.form["method"]
    user_id = int(request.form["user_id"])

    if method == "popularity":
        result = pr.recommend_items(user_id)
    elif method == "content":
        result = cr.recommend_items(user_id)
    elif method == "collaborative":
        result = cfr.recommend_items(user_id)

    print("---------")
    print(result)

 
    return response_success(result)

