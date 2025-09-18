# import necessary libraries
import pandas as pd
import os
import pickle
import random
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

#-------------------------------loading and creating files for dataset 1---------------------------------------------
ratings = pd.read_csv('data/raw/dataset1/ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies_df = pd.read_csv('data/processed/dataset1/movies_cleaned.csv', usecols=['movieId', 'title'])

#------------------------------------Collaborative Filtering Section-----------------------------------------------------------------
# Function to build a pivot table that tracks users activity 
def build_user_item_matrix(ratings):
    """
    Creates a user-item rating matrix for collaborative filtering.
    NaN indicates missing ratings.
    """
    return ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Function to train a SVD model
def train_svd_model(ratings_df, save_path='src/artifacts/svd_model.pkl'):
    """
    Trains an SVD model using Surprise library ans saves it as a pickle file.
    """
    reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max())) # getting the scale of all ratings
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader) # Creating a dataset compatible with the model

    trainset, testset = train_test_split(data, test_size=0.2, random_state=23) # splitting the data
    svd_model = SVD(random_state=23) # initialize the model
    svd_model.fit(trainset)
    predictions = svd_model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)

    # Save the model
    if save_path:
        os.makedirs('artifacts', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(svd_model, f)
        print(f"Model saved to {save_path}")

    return svd_model, predictions, rmse, testset
# to use we run:
svd_model, predictions, rmse, testset = train_svd_model(ratings)

# Function to predict the rating a user will give an unseen movie
def predict_rating(userId, movieId, model=svd_model, verbose=False):
    """
    Predicts the rating a user might give to a movie using a trained Surprise SVD model.
    Parameters:
    - model: Trained Surprise SVD model
    - userId (int): ID of the user
    - movieId (int): ID of the movie
    - verbose (bool): If True, return full prediction object. Default is False.
    Returns:
    - float: estimated rating
    - (optional) full prediction object if verbose=True
    """
    # Ensure userId and movieId are integers
    if not isinstance(userId, int) or not isinstance(movieId, int):
        return "Both userId and movieId must be integers."
     # Prevent predictions for unknown users or movies
    is_known_user = model.trainset.knows_user(model.trainset.to_inner_uid(userId)) if userId in model.trainset._raw2inner_id_users else False
    is_known_movie = model.trainset.knows_item(model.trainset.to_inner_iid(movieId)) if movieId in model.trainset._raw2inner_id_items else False

    if not is_known_user:
        return f"User {userId} not in training data (cold user)."
    if not is_known_movie:
        return f"There is no Movie with ID {movieId}."
    pred = model.predict(userId, movieId)
    if verbose == True:
        return pred
    else:
        return pred.est

# Function to recommend movies based on user activity
def recommend_for_user(userId, model=svd_model, ratings=ratings, movies_df=movies_df,shuffle=False, top_k=10):
    """
    Recommend top_k movies the user hasn't rated yet.
    Parameters:
        userId (int): ID of the user to recommend for.
        model: Trained Surprise SVD model.
        ratings: DataFrame with 'userId', 'movieId', 'rating'.
        movies_df: DataFrame with at least 'movieId' and 'title'.
        top_k (int): Number of recommendations to return.
    Returns:
        DataFrame of top_k recommended movies with average ratings.
    """
    # For new users, recomend popular movies
    if userId not in ratings['userId'].unique():
        print(f"User {userId} is new. Recommending popular movies...")
        # Recommend top-rated movies 
        movie_stats = ratings.groupby('movieId').agg(
            rating_count=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()
        # Filter to movies with enough ratings and recommend at random
        popular_movies = movie_stats[movie_stats['rating_count'] >= 50]
        top_pool = popular_movies.sort_values(by='avg_rating', ascending=False).head(120)
        
        if shuffle==True:
            top_movies = top_pool.sample(n=min(top_k, len(top_pool)))
        else:
            top_movies = top_pool.head(top_k)
        return movies_df[movies_df['movieId'].isin(top_movies['movieId'])][['movieId', 'title']].merge(
            top_movies[['movieId', 'avg_rating']], on='movieId'
        ).sort_values(by='avg_rating', ascending=False).reset_index(drop=True)
    
    # For existing users, recommend unseen movies
    watched = set(ratings[ratings['userId'] == userId]['movieId'].tolist())
    unseen_movies = [mid for mid in movies_df['movieId'].unique() if mid not in watched]
    # Predict once for all unseen movies
    all_predictions = {
        mid: model.predict(userId, mid).est for mid in unseen_movies
    }
    # Build candidate pool from top-N predictions for each watched movie
    candidate_movies = set()
    top_similars = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:int(0.1 * len(unseen_movies))] # Returns the top 10% after sorting
    candidate_movies.update([mid for mid, _ in top_similars])
    if not candidate_movies:
        return pd.DataFrame(columns=['movieId', 'title', 'avg_rating'])

    # Shuffle to vary results slightly
    candidate_movies = list(candidate_movies)
    random.shuffle(candidate_movies)
    # Final recommendation list, already predicted
    final_recommendations = [(mid, all_predictions[mid]) for mid in candidate_movies]
    final_recommendations = sorted(final_recommendations, key=lambda x: x[1], reverse=True)

    if shuffle==True:
        top_recs = pd.DataFrame(random.sample(final_recommendations, k=min(top_k,len(final_recommendations))), columns=['movieId', 'avg_rating'])
    else:
        top_recs = pd.DataFrame(final_recommendations[:top_k], columns=['movieId', 'avg_rating'])
    return top_recs.merge(movies_df[['movieId', 'title']], on='movieId')[['movieId', 'title', 'avg_rating']].sort_values(by='avg_rating', ascending=False)

