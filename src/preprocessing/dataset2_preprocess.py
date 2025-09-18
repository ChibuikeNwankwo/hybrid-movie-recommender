# importing necessary libraries
import pandas as pd
import numpy as np
import pickle
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#-------------------------------loading and creating functions for cleaning dataset 2---------------------------------------------
movies_2 = pd.read_csv('data/processed/dataset2/cleaned_movies2.csv')

# Fix stringified lists to actual lists
def parse_keywords(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception as e:
        return []
for i in ['keywords','cast_names']:
    movies_2[i] = movies_2[i].apply(parse_keywords)

# function to clean text
def clean_text(text):
    """
    Cleans input text by lowering case, removing non-alphabet characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def check_tagline_overlap(df):
    """
    Checks word-level overlap between `overview` and `tagline` per movie.
    Adds a new column `tagline_overlap` representing % of tagline words found in overview.
    
    Returns:
        pd.Series: overlap percentages
    """
    overlaps = []

    for idx, row in df.iterrows():
        overview = clean_text(row.get("overview", ""))
        tagline = clean_text(row.get("tagline", ""))
        
        if not tagline:
            overlaps.append(np.nan)
            continue
        
        overview_words = set(overview.split())
        tagline_words = set(tagline.split())
        
        if not tagline_words:
            overlaps.append(0)
        else:
            common_words = tagline_words & overview_words
            overlap_percent = len(common_words) / len(tagline_words)
            overlaps.append(overlap_percent)

    df["tagline_overlap"] = overlaps
    return df

def prepare_combined_features_basic(df):
    """
    Combines `keywords`, `overview`, and `tagline` into a single string feature.
    Used for text-based similarity in content-based recommenders.
    
    Returns:
        df with new column `combined_tags`
    """
    def combine(row):
        return " ".join([
        " ".join([str(x) for x in row.get("keywords", []) if pd.notna(x)]),
        str(row.get("overview", "")) if pd.notna(row.get("overview", "")) else "",
        str(row.get("tagline", "")) if pd.notna(row.get("tagline", "")) else ""
    ])

    df["combined_tags"] = df.apply(combine, axis=1)
    return df

def prepare_combined_features_extended(df):
    """
    Extends `combined_tags` by including `cast` and `director` to form `combined_tags_extended`.
    Used for testing impact of these features.
    
    Returns:
        df with new column `combined_tags_extended`
    """
    def combine(row):
        return " ".join([
        " ".join([str(x) for x in row.get("keywords", []) if pd.notna(x)]),
        str(row.get("overview", "")) if pd.notna(row.get("overview", "")) else "",
        str(row.get("tagline", "")) if pd.notna(row.get("tagline", "")) else "",
        " ".join([str(x) for x in row.get("cast_names", []) if pd.notna(x)]),
        str(row.get("director", "")) if pd.notna(row.get("director", "")) else ""
        ])
    
    df["combined_tags_extended"] = df.apply(combine, axis=1)
    return df

# function to convert stringified list to actual list
def process_genres(g):
        try:
            genres = ast.literal_eval(g)
            return " ".join([genre.replace(" ", "") for genre in genres])  # remove spaces in names like "Science Fiction"
        except:
            return ""

# running the function on the dataset
m2 = check_tagline_overlap(movies_2)
without_cd = prepare_combined_features_basic(m2)
without_cd['genres'] = without_cd['genres'].apply(process_genres)     

# dropping columns not needed
without_cd = without_cd.drop(columns=['overview', 'tagline', 'keywords'])

#-------------------------------functions for feature preparation-----------------------------------------------------
def build_similarity_matrix(df, text_col='combined_tags'):
    """
    Build a cosine similarity matrix from the given text column of a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_col (str): Column name containing the combined tag string.
        max_features (int): Maximum number of features to include in the vectorizer.
        ngram_range (tuple): The n-gram range for vectorization.

    Returns:
        similarity (ndarray): Cosine similarity matrix.
        vectorizer (CountVectorizer): The fitted vectorizer for inspection.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col} not found in DataFrame.")

    vectorizer = TfidfVectorizer(ngram_range=(1,6), max_features=1000, min_df=2, stop_words='english')
    vectors = vectorizer.fit_transform(df[text_col].fillna(''))
    similarity = cosine_similarity(vectors)
    return similarity, vectorizer

def build_genre_similarity_matrix(df):
    """
    Builds a cosine similarity matrix based on genres only.

    Args:
        df (pd.DataFrame): DataFrame with a 'genres' column.

    Returns:
        np.ndarray: Cosine similarity matrix of shape (n_movies, n_movies)
    """

    df['genres'] = df['genres'].fillna("[]")

    # Vectorize
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    genre_matrix = tfidf.fit_transform(df['genres'])

    # Cosine similarity
    genre_sim_matrix = cosine_similarity(genre_matrix)

    return genre_sim_matrix, tfidf

# running the functions
genre_sim_matrix, tfidf = build_genre_similarity_matrix(without_cd)
tag_sim_matrix, vectorizer = build_similarity_matrix(without_cd, text_col='combined_tags')

#-------------------------------saving the created matrices and dataset-------------------------------------------------------------------
with open('src/artifacts/genre_sim_matrix.pkl', 'wb') as f:
    pickle.dump(genre_sim_matrix,f)

with open('src/artifacts/tag_sim_matrix.pkl', 'wb') as f:
    pickle.dump(tag_sim_matrix,f)

without_cd.to_csv('data/processed/dataset2/final_movies2.csv', index=False)

#-------------------------------function to evaluate the different tags make up---------------------------------------------
def evaluate_variant(df, tags_column, sample_indices):
    tfidf_matrix = vectorizer.fit_transform(df[tags_column])
    cosine_sim = cosine_similarity(tfidf_matrix)

    scores = []
    for idx in sample_indices:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # top-10 (exclude self)
        avg_sim = np.mean([score for _, score in sim_scores])
        scores.append(avg_sim)

    return np.mean(scores)

"""
This function was used to evaluate the similarity scores of the tag matrix with and without the adding the cast and director of the movie to the tags. The evaluation process was not included here.
"""