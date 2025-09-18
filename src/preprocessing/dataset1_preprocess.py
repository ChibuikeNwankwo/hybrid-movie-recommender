# import necessary libraries
import pandas as pd
import os
import ast
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

#-------------------------------loading files and carrying out final cleaning on dataset 1---------------------------------------------
movies = pd.read_csv('data/processed/dataset1/full_movies_cleaned.csv')
ratings = pd.read_csv('data/raw/dataset1/ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies_df = pd.read_csv('data/processed/dataset1/movies_cleaned.csv', usecols=['movieId', 'title'])
# dropping movies without a release year
movies = movies[movies['release_year'].notnull()]

# getting the number of ratings and the average
movies_stats = movies.groupby('movieId')['rating'].agg(rating_count='count', rating_mean='mean').reset_index()
# merging the stats with the titles of each movie for inspection
uniq = movies.copy().drop_duplicates(subset=['movieId'])[['movieId', 'title']]
uniq = movies_stats.merge(uniq, on='movieId', how='left')
# Setting the index to movieId to align
uniq.set_index("movieId", inplace=True)

#--------------------------------preprocessing features------------------------
# Function to preprocess features and create binary columns for genres and tags
def preprocess_features(movies_df):
    """
    Converts 'genres_list' and 'tags_list' columns into binary feature columns using MultiLabelBinarizer.
    Args:
        movies_df (pd.DataFrame): DataFrame containing movie data with 'genres_list' and 'tags_list' columns.
    Returns:
        - updated movies_df with binary columns
        - combined_matrix of encoded features (indexed by movieId)
        - dictionary of fitted encoders
    """
    encoders = {} # Dictionary to store fitted MultiLabelBinarizer instances
    for col in ['genres_list', 'tags_list']:
        # Convert stringified lists to actual lists 
        movies_df[col] = movies_df[col].apply(ast.literal_eval)

        # Fit MultiLabelBinarizer and transform the list column
        mlb = MultiLabelBinarizer()
        binary_matrix = pd.DataFrame(
            mlb.fit_transform(movies_df[col]),
            columns=[f"{col}_{cls}" for cls in mlb.classes_],
            index=movies_df.index
        )
        # Append binary columns to movies_df
        movies_df = pd.concat([movies_df, binary_matrix], axis=1)
        encoders[col] = mlb

    # Extract binary feature column names using correct prefixes
    genre_cols = [col for col in movies_df.columns if col.startswith("genres_")]
    tag_cols = [col for col in movies_df.columns if col.startswith("tags_")]

    # Validate that encoded columns were added
    if not genre_cols and not tag_cols:
        raise ValueError("No encoded genre or tag columns found. Check preprocessing steps.")

    # Create final feature matrix
    combined_matrix = movies_df.set_index("movieId")[genre_cols * 2 + tag_cols * 1]

    return movies_df, combined_matrix, encoders

# Run
movies, combined_matrix, encoders = preprocess_features(movies)

# Drop object-type columns
combined_matrix_cleaned = combined_matrix.drop(columns=combined_matrix.select_dtypes(include='object').columns)
# Ensure everything is numeric
combined_matrix_cleaned = combined_matrix_cleaned.apply(pd.to_numeric, errors='coerce').fillna(0)
# Remove duplicate indices
combined_matrix_cleaned = combined_matrix_cleaned[~combined_matrix_cleaned.index.duplicated(keep='first')]
# Merge combined_matrix with movies_stats(uniq)
combined_with_stats = combined_matrix_cleaned.join(uniq, how="left")

# setting up the weights for each feature
GENRE_WEIGHT = 0.2
TAG_WEIGHT = 0.35
VOTE_WEIGHT = 0.2
RATING_WEIGHT = 0.25
# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Apply scaling
scaled_values = scaler.fit_transform(
    combined_with_stats[['rating_count', 'rating_mean']].fillna(0))

# Add scaled values to the DataFrame
combined_with_stats['scaled_vote_count'] = scaled_values[:, 0]
combined_with_stats['scaled_avg_rating'] = scaled_values[:, 1]
# Identify the one-hot columns for genres and tags
genre_cols = [col for col in combined_with_stats.columns if col.startswith('genres_')]
tag_cols = [col for col in combined_with_stats.columns if col.startswith('tags_')]
# Apply weights
weighted_genres = combined_with_stats[genre_cols] * GENRE_WEIGHT
weighted_tags = combined_with_stats[tag_cols] * TAG_WEIGHT

# Final Matrix
final_matrix = pd.concat([
    weighted_genres,
    weighted_tags,
    combined_with_stats[['scaled_vote_count']].mul(VOTE_WEIGHT),
    combined_with_stats[['scaled_avg_rating']].mul(RATING_WEIGHT)
], axis=1)
# Save as a csv file for easy reloading
combined_with_stats.to_csv('data/processed/dataset1/combined_with_stats.csv')
# Compute cosine similarity matrix from combined features
cosine_sim_matrix = cosine_similarity(final_matrix)
cosine_sim_df = pd.DataFrame(
    cosine_sim_matrix, 
    index=combined_with_stats.index,  # movieId
    columns=combined_with_stats.index  # movieId
)
# Ensure the cosine similarity matrix is symmetric
cosine_sim_df = (cosine_sim_df + cosine_sim_df.T) / 2
# Save the cosine similarity matrix to a pickle file
if not os.path.exists('src/artifacts'):
    os.makedirs('src/artifacts') # Create artifacts directory if it doesn't exist
with open('src/artifacts/cosine_sim_matrix.pkl', 'wb') as f:
    pickle.dump(cosine_sim_df, f)
