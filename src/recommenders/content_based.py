# import necessary libraries
import pandas as pd
import numpy as np
import pickle
from difflib import get_close_matches

#-------------------------------loading files from dataset 1 -------------------------------
movies = pd.read_csv('data/processed/dataset1/full_movies_cleaned.csv')
movies_unique = movies.copy().drop_duplicates(subset='movieId').drop(columns=['genres_list', 'tags_list'])
with open('src/artifacts/cosine_sim_matrix.pkl', 'rb') as f:
   cosine_sim_matrix = pickle.load(f)

#-------------------------------loading files from dataset 2-------------------------------
movies_df2 = pd.read_csv('data/processed/dataset2/final_movies2.csv')
intersection_df = pd.read_csv('data/intersection/intersection.csv')

with open('src/artifacts/genre_sim_matrix.pkl', 'rb') as f:
   genre_sim_matrix = pickle.load(f)

with open('src/artifacts/tag_sim_matrix.pkl', 'rb') as f:
    tag_sim_matrix = pickle.load(f)

# index order must match the rows/columns of the similarity matrices
movie_ids = movies_df2['id'].tolist()
id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
idx_to_id = {idx: movie_id for movie_id, idx in id_to_idx.items()}

#-------------------------------------Content Based Filtering Section------------------------------------------------------------------
# Function to recommend similar movies based on movieId
def recommend_similar_movies_by_id(movie_id, cosine_sim_df=cosine_sim_matrix, movies_df=movies_unique, top_k=10):
    """
    Recommend top_k movies similar to a given movieId using cosine similarity.
    Args:
        movie_id (int): The reference movie ID.
        cosine_sim_df (pd.DataFrame): DataFrame of cosine similarity scores (movieId x movieId).
        movies_df (pd.DataFrame): Contains movie metadata
        top_k (int): Number of similar movies to return.
    Returns:
        pd.DataFrame: Top_k recommended movies with similarity scores.    
    """
    # Check if movie_id exists in the similarity matrix
    if movie_id not in cosine_sim_df.index:
        raise ValueError(f"MovieId {movie_id} not found in similarity_matrix.")
    # Get similarity scores for the given movieId and drop the movie itself
    similarity_series = cosine_sim_df.loc[movie_id].drop(index=movie_id)
    # Sort by similarity score
    most_similar = similarity_series.sort_values(ascending=False)
    # Get top movieIds
    top_movie_ids = most_similar.head(30).index

    # Join with the movies dataframe to get the meta data
    recommended = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId','title']].copy()
    # Add similarity scores column 
    recommended['similarity'] = recommended['movieId'].map(most_similar)
    # sort and return
    return recommended.sort_values(by='similarity', ascending=False).sample(n=top_k).sort_values(by='similarity', ascending=False).reset_index(drop=True)

# Function to recommend similar movies from dataset 2 based on movieId
def recommend_similar_movies_by_id_dataset2(
    movie_id,
    tags_sim_matrix=tag_sim_matrix,
    genres_sim_matrix=genre_sim_matrix,
    id_to_idx=id_to_idx,
    idx_to_id=idx_to_id,
    movies_df=movies_df2,
    weight_tags=0.6,
    weight_genres=0.4,
    top_k=10,
    sort='similarity'
):
    """
    Recommend top_k movies similar to a given movie ID in Dataset 2 using NumPy cosine similarity arrays.
    Args:
        movie_id (int): Movie ID from Dataset 2.
        tags_sim_matrix (np.ndarray): Cosine sim array for tags (N x N).
        genres_sim_matrix (np.ndarray): Cosine sim array for genres (N x N).
        id_to_idx (dict): Maps movie_id → index in sim matrices.
        idx_to_id (dict): Maps index → movie_id.
        movies_df (pd.DataFrame): Dataset 2 movie DataFrame with 'id', 'title', etc.
        weight_tags (float): Weight for tag similarity.
        weight_genres (float): Weight for genre similarity.
        top_k (int): Number of recommendations to return.
        sort (str): 'similarity', 'popular', or 'random'.
    Returns:
        pd.DataFrame: Top_k recommended movies.
    """
    if movie_id not in id_to_idx:
        raise ValueError(f"Movie ID {movie_id} not found in similarity index.")
    idx = id_to_idx[movie_id]
    N = tags_sim_matrix.shape[0]

    # Get similarity vectors and compute weighted combination
    tag_scores = tags_sim_matrix[idx]
    genre_scores = genres_sim_matrix[idx]
    combined_scores = (weight_tags * tag_scores) + (weight_genres * genre_scores)

    # Remove the movie itself
    combined_scores[idx] = -1
    # Get top N indices
    top_indices = combined_scores.argsort()[::-1][:30]
    # Map back to movie IDs
    top_movie_ids = [idx_to_id[i] for i in top_indices]
    # Get metadata
    recommended = movies_df[movies_df['id'].isin(top_movie_ids)].copy()
    recommended['similarity'] = recommended['id'].map(
        {idx_to_id[i]: combined_scores[i] for i in top_indices}
    )
    # Add popularity score
    recommended['popularity_score'] = recommended['vote_count'] * recommended['vote_average']
    # Final sort
    if sort == 'popular':
        recommended = recommended.sort_values(by='popularity_score', ascending=False)
    elif sort == 'random':
        recommended = recommended.sample(n=min(top_k, len(recommended)))
    else:
        recommended = recommended.sort_values(by='similarity', ascending=False)

    return recommended[['id','title','similarity']].head(top_k).reset_index(drop=True)

# Function to recommend movies based on a partial title match from both dataset 1 and 2
def unified_recommend_by_title(title, top_k=10):
    """
    Generate unified movie recommendations across Dataset 1 and Dataset 2 by title search.

    This function searches both datasets for movies whose titles contain the given
    input string (case-insensitive). For each matched movie:
      - Recommendations from Dataset 1 are retrieved via collaborative/content filtering,
        with similarity scores down-weighted (multiplied by 0.78) if only present there.
      - Recommendations from Dataset 2 are retrieved using content-based filtering
        (tag and genre similarity matrices).
    All recommendations are combined, duplicates across datasets are resolved using the
    intersection mapping, and the final list is ranked by similarity.

    Args:
        title (str): Partial or full movie title to search for.
        top_k (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pd.DataFrame or None: A DataFrame with columns ['title', 'similarity']
        containing up to top_k recommended movies, sorted by similarity score.
        Returns None if no matches were found in either dataset.
    """
    title = title.lower().strip()
    # Match in both datasets
    matched1 = movies_unique[movies_unique['title'].str.lower().str.contains(title, na=False)]
    matched2 = movies_df2[movies_df2['title'].str.lower().str.contains(title, na=False)]

    if matched1.empty and matched2.empty:
        return None
    matched_ids1 = matched1['movieId'].drop_duplicates().tolist()
    matched_ids2 = matched2['id'].drop_duplicates().tolist()

    all_recs = []
    # Dataset 1 recommendations
    for movie_id in matched_ids1:
        recs = recommend_similar_movies_by_id(movie_id, top_k=top_k)
        recs['similarity'] *= 0.78 
        all_recs.extend(recs.to_dict(orient='records'))
    # Dataset 2 recommendations
    for movie_id in matched_ids2:
        recs = recommend_similar_movies_by_id_dataset2(
            movie_id,
            tags_sim_matrix=tag_sim_matrix,
            genres_sim_matrix=genre_sim_matrix,
            top_k=top_k
        )
        all_recs.extend(recs.to_dict(orient='records'))

    # Remove duplicates across datasets using the intersection
    seen_ids = set()
    unique_recs = []
    for rec in all_recs:
        global_id = None
        if 'movieId' in rec:
            global_id = rec['movieId']
        elif 'id' in rec:
            match = intersection_df[intersection_df['tmdbId'] == rec['id']]
            if not match.empty:
                global_id = match['movieId'].values[0]
            else:
                global_id = f"ds2_{rec['id']}"

        if global_id not in seen_ids:
            seen_ids.add(global_id)
            unique_recs.append(rec)
    df = pd.DataFrame(unique_recs)
    df = df[['title', 'similarity']].sort_values(by='similarity', ascending=False).head(top_k).reset_index(drop=True)
    return df

# Function to return movies title based on a partial title match from both dataset 1 and 2
def unified_search_by_title(query, top_k=30):
    """
    Search for movies by partial title across Dataset 1 and Dataset 2.

    This function performs a case-insensitive search for titles containing the
    query string in both datasets. Results are standardized into a common schema,
    merged, and deduplicated by title. Ratings are normalized so that Dataset 1
    ratings are rescaled to align with Dataset 2 scale (multiplied by 2).
    The function returns the top-k matches sorted by rating.

    Args:
        query (str): Partial or full movie title to search for (case-insensitive).
        top_k (int, optional): Maximum number of results to return. Defaults to 30.

    Returns:
        pd.Series or None: A Series of movie titles (up to top_k), sorted by rating.
        Returns None if no matches were found in either dataset.
    """
    query = query.lower().strip()
    # Case-insensitive partial match in both datasets
    matched1 = movies_unique[movies_unique['title'].str.lower().str.contains(query, na=False)].copy()
    matched2 = movies_df2[movies_df2['title'].str.lower().str.contains(query, na=False)].copy()
    if matched1.empty and matched2.empty:
        return None

    # Add source column for traceability
    matched1['source'] = 'dataset1'
    matched2['source'] = 'dataset2'
    # Normalize column names
    matched1 = matched1.rename(columns={'movieId': 'id', 'rating': 'rating'})
    matched2 = matched2.rename(columns={'id': 'id', 'vote_average': 'rating'})  # Adjust if your column is different
    # Select only relevant columns
    matched1 = matched1[['title', 'rating', 'source']]
    matched1['rating'] = matched1['rating']*2
    matched2 = matched2[['title', 'rating', 'source']]

    # Combine both
    combined = pd.concat([matched1, matched2], ignore_index=True)
    # Drop duplicates by title (keeping the highest rating)
    combined = combined.sort_values(by='rating', ascending=False)
    combined = combined.drop_duplicates(subset='title', keep='first')
    # Return top-k sorted by rating
    combined = combined.sort_values(by='rating', ascending=False).head(top_k).reset_index(drop=True)
    return combined['title']

# Function to recommend movies based on genres from both dataset 1 and 2
def unified_recommend_by_genres(genres, top_k=10, seed='random', mode='AND'):
    """
    Recommend movies based on genres using both datasets.
    Parameters:
    - genres: str or list of partial genre names
    - top_k: number of recommendations
    - seed: 'popular' or 'random' seed movie
    - mode: 'AND' or 'OR' for genre filtering
    Returns:
    - DataFrame with top-k recommended movies
    """
    if isinstance(genres, str):
        genres = [genres]
    genres = [g.strip().lower() for g in genres]
    # Match genre columns in Dataset 1 (one-hot encoded)
    genre_cols_ds1 = [col for col in movies_unique.columns if col.startswith("genres_list_")]
    matched_cols_ds1 = list({
        col for g in genres for col in genre_cols_ds1 if g in col.lower()
    })
    # Match genre presence in Dataset 2
    movies_df2['genres_list'] = movies_df2['genres'].str.split()
    movies_df2['genres_list'] = movies_df2['genres_list'].apply(
    lambda x: x if isinstance(x, list) else [])
    
    matched_mask_ds2 = np.zeros(len(movies_df2), dtype=bool)
    for g in genres:
        matched_mask_ds2 |= movies_df2['genres_list'].apply(
            lambda glist: any(g in str(genre).lower() for genre in glist)
        )
    # Filter Dataset 1
    filtered_ds1 = movies_unique.copy()
    if matched_cols_ds1:
        if mode.upper() == 'AND':
            for col in matched_cols_ds1:
                filtered_ds1 = filtered_ds1[filtered_ds1[col] == 1]
        elif mode.upper() == 'OR':
            filtered_ds1 = filtered_ds1[filtered_ds1[matched_cols_ds1].sum(axis=1) >= 1]
    # Filter Dataset 2
    filtered_ds2 = movies_df2[matched_mask_ds2]
    # Seed movie from each dataset
    seed_movie_ds1 = None
    seed_movie_ds2 = None

    if not filtered_ds1.empty:
        seed_movie_ds1 = (
            filtered_ds1.sort_values('rating_count', ascending=False).head(7).sample(1).iloc[0]
            if seed == 'popular'
            else filtered_ds1.sample(1).iloc[0]
        )
    if not filtered_ds2.empty:
        seed_movie_ds2 = (
            filtered_ds2.sort_values('vote_count', ascending=False).head(7).sample(1).iloc[0]
            if seed == 'popular'
            else filtered_ds2.sample(1).iloc[0]
        )
    # Collect recommendations
    recs_ds1 = pd.DataFrame()
    recs_ds2 = pd.DataFrame()

    if seed_movie_ds1 is not None:
        recs_ds1 = recommend_similar_movies_by_id(
            seed_movie_ds1['movieId'],
            top_k=top_k
        )
        recs_ds1['similarity'] *= 0.78
        recs_ds1['source'] = 'ds1'
    if seed_movie_ds2 is not None:
        recs_ds2 = recommend_similar_movies_by_id_dataset2(
            seed_movie_ds2['id'],
            tags_sim_matrix=tag_sim_matrix,
            genres_sim_matrix=genre_sim_matrix,
            top_k=top_k
        )
        recs_ds2['source'] = 'ds2'
    if recs_ds1.empty and recs_ds2.empty:
        return f"No movies found for genres: {genres}"

    # Rename IDs for merging
    recs_ds1 = recs_ds1.rename(columns={'movieId': 'unified_id'})
    recs_ds2 = recs_ds2.rename(columns={'id': 'unified_id'})
    # Map Dataset 1 movieIds to Dataset 2 ids if available
    if not intersection_df.empty:
        map_ds1_to_ds2 = intersection_df.set_index('movieId')['tmdbId'].to_dict()
        recs_ds1['unified_id'] = recs_ds1['unified_id'].map(map_ds1_to_ds2).fillna(recs_ds1['unified_id'])
    # Merge, deduplicate
    combined_recs = pd.concat([recs_ds1, recs_ds2], ignore_index=True)
    combined_recs = combined_recs.drop_duplicates(subset='unified_id', keep='first')

    # Sort and return top-k
    df = combined_recs.sort_values(by='similarity', ascending=False).head(top_k).reset_index(drop=True)
    return df[['unified_id','title']]

# Function to recommend movies based on partial tags and genres(optional)
def recommend_by_tags_and_genres(tags, genres=None, top_k=10, mode='AND', sort='popular'):
    """
    Recommend movies based on tag(s) and optional genre(s).
    Parameters:
    - tags (str or list): tag names (partial matches allowed)
    - genres (str or list): genres (partial match allowed)
    - top_k (int): number of recommendations
    - seed: 'random' or 'popular' for seed movie selection (unused now)
    - mode: 'AND' or 'OR' for combining tag/genre filters (affects tag-genre combo logic)
    - sort: 'popular' (default) or 'random' for selecting top_k from final pool
    Returns:
    - DataFrame with top_k recommended movies
    """
    if tags is None:
        return ("No tags or genres provided. Please specify at least a tag.") # Tells user to specify a tag
    if isinstance(tags, str):
        tags = [tags] # converts entry to a list
    if genres is not None and isinstance(genres, str):
        genres = [genres]

    sort = str(sort).lower().strip()
    if sort not in ['popular', 'random']:
        return f"Invalid sort option: {sort}. Use 'popular' or 'random'." # validates the sort argument
    # Finds all tag and genre columns in the dataframe
    tag_cols = [col for col in movies_unique.columns if col.startswith("tags_list_")]
    genre_cols = [col for col in movies_unique.columns if col.startswith("genres_list_")]

    # Match tag columns via partial search
    matched_tag_cols = []
    for tag in tags:
        tag = tag.strip().lower()
        matches = [col for col in tag_cols if tag in col.lower()]
        if not matches:
            suggestions = get_close_matches(tag, tag_cols, n=3, cutoff=0.6)
            return f"Tag '{tag}' not found. Did you mean: {', '.join(suggestions)}?"
        matched_tag_cols.extend(matches)

    matched_tag_cols = list(set(matched_tag_cols))  # removes duplicate
    # Match genre columns via partial search
    matched_genre_cols = []
    if genres:
        for genre in genres:
            genre = genre.strip().lower()
            matches = [col for col in genre_cols if genre in col.lower()]
            if not matches:
                suggestions = get_close_matches(genre, genre_cols, n=3, cutoff=0.6)
                return f"Genre '{genre}' not found. Did you mean: {', '.join(suggestions)}?"
            matched_genre_cols.extend(matches)

        matched_genre_cols = list(set(matched_genre_cols))

    # Build all tag-genre combos
    combo_filters = []
    for tag_col in matched_tag_cols:
        if not genres:
            combo = movies_unique[movies_unique[tag_col] == 1]
        else:
            for genre_col in matched_genre_cols:
                combo = movies_unique[(movies_unique[tag_col] == 1) & (movies_unique[genre_col] == 1)]
                if not combo.empty:
                    combo_filters.append(combo)

        if not genres and not combo.empty:
            combo_filters.append(combo)

    if not combo_filters:
        return f"No movies found with tags: {tags} and genres: {genres or 'None'} in {mode} mode."
    # Pool all matching movies together
    pooled_movies = pd.concat(combo_filters).drop_duplicates(subset='movieId')
    if pooled_movies.empty:
        return f"No movies found after pooling tag-genre matches."

    # Select top_k movies
    if sort == 'popular' and 'vote_count' in pooled_movies.columns:
        selected = pooled_movies.sort_values('vote_count', ascending=False).head(top_k)
    else:
        selected = pooled_movies.sample(min(top_k, len(pooled_movies)))
    return selected[['movieId', 'title']]

