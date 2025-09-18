# import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from .collaborative import predict_rating, recommend_for_user
from .content_based import recommend_similar_movies_by_id

#-----------------------------------loading files -------------------------------------------------------------
movies = pd.read_csv('data/processed/dataset1/full_movies_cleaned.csv')
ratings = pd.read_csv('data/raw/dataset1/ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies_unique = movies.copy().drop_duplicates(subset='movieId').drop(columns=['genres_list', 'tags_list'])
combined_with_stats= pd.read_csv('data/processed/dataset1/combined_with_stats.csv')

movies_df2 = pd.read_csv('data/processed/dataset2/final_movies2.csv')
intersection_df = pd.read_csv('data/processed/dataset2/intersection.csv')

with open('src/artifacts/genre_sim_matrix.pkl', 'rb') as f:
   genre_sim_matrix = pickle.load(f)

with open('src/artifacts/tag_sim_matrix.pkl', 'rb') as f:
    tag_sim_matrix = pickle.load(f)
#-----------------------------------creating files to be used for the hybrid model-----------------------------
# Generate intersection_map: Dataset1 movieId → Dataset2 id
intersection_map = dict(zip(intersection_df['movieId'], intersection_df['tmdbId']))
dataset2_only = movies_df2[~movies_df2['id'].isin(intersection_df['tmdbId'])].copy()
dataset2_title_df = movies_df2[['id', 'title']]

# creating a copy to avoid mutation
movie_df2_filtered = movies_df2.reset_index(drop=True).copy()  
# Get the ordered list of movie IDs used in the matrix
ordered_ids = movie_df2_filtered['id'].tolist()
# movieId → index (for lookup in sim matrix)
movieId_index_map = {movie_id: idx for idx, movie_id in enumerate(ordered_ids)}
# index → movieId (for inverse lookup)
index_movieId_map = {idx: movie_id for movie_id, idx in movieId_index_map.items()}
 
user_stats = movies.groupby('userId')['movieId'].agg(movie_count='count').reset_index()

#----------------------------------------------------Hybrid Model---------------------------------------------------------------------
# Function to get user type based on activity
def get_user_type_weights(userId, df=user_stats):
    """
    Returns a dictionary of weights for different scoring components based on user type.
    Args:
        user_type (str): Type of user (e.g., 'binge_watcher', 'occasional', 'balanced').
    Returns:
        dict: Weights for CF, CBF, vote_count, and rating_mean scores.
    """
    # weights dict
    weights_by_type = {
        'binge_watcher': {'cf': 0.6, 'cbf': 0.2, 'vc': 0.1, 'rm': 0.1},
        'occasional': {'cf': 0.2, 'cbf': 0.6, 'vc': 0.1, 'rm': 0.1},
        'balanced': {'cf': 0.4, 'cbf': 0.4, 'vc': 0.1, 'rm': 0.1},
        'new_user': {'cf': 0.0, 'cbf': 0.35, 'vc': 0.35, 'rm': 0.3}
    }
    row = df[df['userId'] == userId] # checks if the user exists in the dataset
    if row.empty:
        user_type = 'new_user'
    else:
        user_count = row.iloc[0]['movie_count']
        # Determine user type based on movie count
        if user_count > 168:
            user_type = 'binge_watcher'
        elif user_count > 50:
            user_type = 'balanced'
        else:
            user_type = 'occasional'
    return weights_by_type.get(user_type, weights_by_type[user_type])

# Function to get a user's top rated movies
def get_user_top_movies(user_id, ratings_df=ratings, threshold=4.0):
    """
    Get the list of movieIds the user rated >= threshold.
    Args:
        user_id (int): The user ID.
        ratings_df (pd.DataFrame): DataFrame containing user ratings.
        threshold (float): Minimum rating to consider a movie as 'liked'.
    Returns:
        list: List of liked movieIds.
    """
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= threshold]['movieId'].tolist()
    return liked_movies

# Function to recommend movies for users based on all parameters from dataset 1 and 2
def unified_hybrid_recommender(
    user_id, 
    top_k=20,
    movies_df=movies_unique, 
    combined_with_stats=combined_with_stats,
    sim_genre_matrix=genre_sim_matrix,
    sim_tag_matrix=tag_sim_matrix,
    movieId_index_map=movieId_index_map,
    index_movieId_map=index_movieId_map,
    intersection_df=intersection_df,
    dataset2_titles_df=dataset2_title_df,
    shuffle=False,
    boost_factor=1.2,
    dataset2_only_cap_ratio=0.3
):
    weights = get_user_type_weights(user_id)
    combined_with_stats = combined_with_stats.reset_index()
    # -----------------Dataset 1 CF--------------------------------
    cf_weight = weights.get('cf', 0.0)
    if cf_weight > 0.0:
        try:
            cf_recs = recommend_for_user(user_id, top_k=100)
            cf_recs['cf_score'] = cf_recs['movieId'].apply(lambda x: predict_rating(user_id, x))
            cf_recs = cf_recs[['movieId', 'cf_score']]
            cf_recs['cf_score'] = MinMaxScaler().fit_transform(cf_recs[['cf_score']])
        except Exception:
            cf_recs = pd.DataFrame(columns=['movieId', 'cf_score'])
    else:
        cf_recs = pd.DataFrame(columns=['movieId', 'cf_score'])

    # --------------------Dataset 1 CBF----------------------------------
    liked_movies = get_user_top_movies(user_id)
    if liked_movies:
        cbf_recs = pd.DataFrame()
        for mid in liked_movies:
            try:
                recs = recommend_similar_movies_by_id(mid, top_k=20)
                cbf_recs = pd.concat([cbf_recs, recs], axis=0)
            except Exception:
                continue
        cbf_recs = cbf_recs.groupby('movieId', as_index=False)['similarity'].max()
        cbf_recs.rename(columns={'similarity': 'cbf_score'}, inplace=True)
        cbf_recs['cbf_score'] = MinMaxScaler().fit_transform(cbf_recs[['cbf_score']])
    else:
        fallback_pool = combined_with_stats.sort_values(by='rating_count', ascending=False).head(200).copy()
        user_seed = abs(hash(user_id)) % (2**32)
        if shuffle:
            fallback_recs = fallback_pool.sample(n=min(top_k, len(fallback_pool)))
        else:
            fallback_recs = fallback_pool.sample(n=min(top_k, len(fallback_pool)), random_state=user_seed)
        cbf_recs = fallback_recs[['movieId']].copy()
        cbf_recs['cbf_score'] = 1.0

    # ------------------------Merge Dataset 1----------------------------
    merged = pd.merge(cf_recs, cbf_recs, on='movieId', how='outer')
    merged = pd.merge(merged, combined_with_stats[['movieId', 'rating_count', 'rating_mean']], on='movieId', how='left')
    for col in ['cf_score', 'cbf_score', 'rating_count', 'rating_mean']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    merged['vc_score'] = MinMaxScaler().fit_transform(merged[['rating_count']])
    merged['rm_score'] = MinMaxScaler().fit_transform(merged[['rating_mean']])
    merged['final_score'] = (
        weights['cf'] * merged['cf_score'] +
        weights['cbf'] * merged['cbf_score'] +
        weights['vc'] * merged['vc_score'] +
        weights['rm'] * merged['rm_score']
    )
    merged = pd.merge(merged, movies_df[['movieId', 'title']], on='movieId', how='left')

    # ----------------------Build User Profile Vectors-------------------------
    profile_indices = [
        movieId_index_map.get(intersection_df.loc[intersection_df['movieId'] == mid, 'tmdbId'].values[0])
        for mid in liked_movies
        if mid in intersection_df['movieId'].values and movieId_index_map.get(intersection_df.loc[intersection_df['movieId'] == mid, 'tmdbId'].values[0]) is not None
    ]
    if profile_indices:
        user_tag_profile = np.mean(sim_tag_matrix[profile_indices], axis=0)
        user_genre_profile = np.mean(sim_genre_matrix[profile_indices], axis=0)
    else:
        user_tag_profile = np.zeros(sim_tag_matrix.shape[1])
        user_genre_profile = np.zeros(sim_genre_matrix.shape[1])

    # -----------------------------Dataset 2-Only (CBF)---------------------------
    dataset2_only_ids = set(movieId_index_map) - set(intersection_df['tmdbId'])
    dataset2_only_recs = []
    for movie2_id in dataset2_only_ids:
        idx = movieId_index_map[movie2_id]
        cbf_score = 0.0
        if user_tag_profile.any() or user_genre_profile.any():
            tag_sim = sim_tag_matrix[idx]
            genre_sim = sim_genre_matrix[idx]
            cbf_score = float(0.6 * cosine_similarity([tag_sim], [user_tag_profile])[0][0] +
                              0.4 * cosine_similarity([genre_sim], [user_genre_profile])[0][0])
            if cbf_score < 0.01:
                cbf_score = np.random.uniform(0.01, 0.05)
        dataset2_only_recs.append({
            'movie2_id': movie2_id,
            'cbf_score': cbf_score,
            'cf_score': 0.0,
            'vc_score': 0.0,
            'rm_score': 0.0
        })
    dataset2_only_df = pd.DataFrame(dataset2_only_recs)
    if not dataset2_only_df.empty:
        dataset2_only_df['cbf_score'] = MinMaxScaler().fit_transform(dataset2_only_df[['cbf_score']])
        dataset2_only_df['cbf_score'] += np.random.normal(0, 0.005, len(dataset2_only_df))
        dataset2_only_df['cbf_score'] = dataset2_only_df['cbf_score'].clip(0, 1)
        dataset2_only_df['final_score'] = weights['cbf'] * dataset2_only_df['cbf_score']
        dataset2_only_df = pd.merge(dataset2_only_df, dataset2_titles_df[['id', 'title']], left_on='movie2_id', right_on='id', how='left')
    else:
        dataset2_only_df = pd.DataFrame(columns=['movie2_id', 'cbf_score', 'final_score', 'title'])

    # --------------------------Dataset 2-Intersection------------------------
    dataset2_cbf = pd.DataFrame(columns=['movie2_id', 'cbf_score'])
    for mid in liked_movies:
        if mid in intersection_df['movieId'].values:
            movie2_id = intersection_df.loc[intersection_df['movieId'] == mid, 'tmdbId'].values[0]
            idx = movieId_index_map.get(movie2_id)
            if idx is None:
                continue
            sim_vec = 0.6 * sim_tag_matrix[idx] + 0.4 * sim_genre_matrix[idx]
            top_indices = sim_vec.argsort()[::-1][:top_k]
            for i in top_indices:
                movie2 = index_movieId_map[i]
                score = sim_vec[i]
                dataset2_cbf = pd.concat([dataset2_cbf, pd.DataFrame({
                    'movie2_id': [movie2],
                    'cbf_score': [score]
                })], ignore_index=True)
    if not dataset2_cbf.empty:
        dataset2_cbf = dataset2_cbf.groupby('movie2_id', as_index=False)['cbf_score'].max()
        dataset2_cbf['cbf_score'] = MinMaxScaler().fit_transform(dataset2_cbf[['cbf_score']])
        dataset2_cbf['cf_score'] = 0.0
        dataset2_cbf['vc_score'] = 0.0
        dataset2_cbf['rm_score'] = 0.0
        dataset2_cbf['final_score'] = weights['cbf'] * dataset2_cbf['cbf_score']
        dataset2_cbf = pd.merge(dataset2_cbf, dataset2_titles_df[['id', 'title']], left_on='movie2_id', right_on='id', how='left')
    else:
        sampled_movies = dataset2_titles_df.sample(n=min(100, len(dataset2_titles_df)))
        dataset2_cbf = pd.DataFrame({
            'movie2_id': sampled_movies['id'].values,
            'cbf_score': np.random.uniform(0.05, 0.2, len(sampled_movies))
        })
        dataset2_cbf['cf_score'] = 0.0
        dataset2_cbf['vc_score'] = 0.0
        dataset2_cbf['rm_score'] = 0.0
        dataset2_cbf['final_score'] = weights['cbf'] * dataset2_cbf['cbf_score']
        dataset2_cbf = pd.merge(dataset2_cbf, dataset2_titles_df[['id', 'title']], left_on='movie2_id', right_on='id', how='left')

    # -----------------------Merge All-----------------------------
    merged['source'] = 'dataset1'
    dataset2_cbf['source'] = 'dataset2_intersection'
    dataset2_only_df['source'] = 'dataset2_only'
    all_recs = pd.concat([
        merged[['movieId', 'final_score', 'title', 'source']],
        dataset2_cbf.rename(columns={'movie2_id': 'movieId'})[['movieId', 'final_score', 'title', 'source']],
        dataset2_only_df.rename(columns={'movie2_id': 'movieId'})[['movieId', 'final_score', 'title', 'source']]
    ])
    all_recs = all_recs.drop_duplicates(subset='movieId')

    # ------------------------Boost Dataset2-only if needed----------------------
    is_user_weak = cf_recs.empty and liked_movies == []
    if is_user_weak:
        all_recs.loc[all_recs['source'] == 'dataset2_only', 'final_score'] *= boost_factor

    # --------------------------Cap Dataset2-only in Top-K-----------------
    all_recs = all_recs.sort_values(by='final_score', ascending=False)
    max_d2_only = int(dataset2_only_cap_ratio * top_k)
    d2_only = all_recs[all_recs['source'] == 'dataset2_only']
    if shuffle:
        d2_only = d2_only.sample(frac=1).head(max_d2_only)  # <-- SHUFFLE this pool
    else:
        d2_only = d2_only.head(max_d2_only)
    others_pool = all_recs[all_recs['source'] != 'dataset2_only']

    if shuffle:
        # Sample from top-N of others for variety
        top_pool = others_pool.head(max(3 * (top_k - len(d2_only)), top_k))
        others = top_pool.sample(n=min(top_k - len(d2_only), len(top_pool)))
    else:
        others = others_pool.head(top_k - len(d2_only))

    final_topk = pd.concat([d2_only, others]).sort_values(by='final_score', ascending=False).reset_index(drop=True)
    return final_topk[['movieId','title']]
