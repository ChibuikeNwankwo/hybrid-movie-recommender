# === Importing necessary libraries=== #
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

# === Load Precomputed Functions === #
from src.recommenders.hybrid import unified_hybrid_recommender
from src.recommenders.content_based import unified_recommend_by_genres, unified_recommend_by_title, unified_search_by_title

# === Loading pickle file === #
unique_genre = joblib.load('src/artifacts/unique_genre.pkl') 

app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid + Content-based recommendation system",
    version="1.0"
)
# === ROUTES === #
@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.get("/recommend/hybrid")
def get_hybrid_recommendations(
    user_id: int,
    top_k: int = 10
):
    """
    Hybrid recommendation using collaborative + content-based filtering.
    """
    if user_id < 1 or user_id is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid user ID."
        )
    recommendations = unified_hybrid_recommender(
        user_id=user_id,
        top_k=top_k,
        shuffle=False,    
    )

    return recommendations[["movieId", "title"]].to_dict(orient="records")

@app.get("/recommend/by-title")
def get_recommendations_by_title(
    title: str,
    top_k: int = 10
):
    """
    Content-based recommendation using cosine similarity from a reference movie.
    """
    if not title or not title.strip():
        raise HTTPException(
        status_code=400,
        detail="Title query cannot be empty."
    )
    recommendations = unified_recommend_by_title(
        title=title,
        top_k=top_k
    )
    if recommendations is None or recommendations.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No movie found with the title '{title}'. Please check for typos or try a different movie."
        )
    return recommendations.to_dict(orient="records")

@app.get("/search/by-title")
def get_title_name(
    title: str,
    top_k: int = 10):
    """
    Returns a list of all movies matching the title being searched
    """
    if not title or not title.strip():
        raise HTTPException(
        status_code=400,
        detail="Title query cannot be empty."
    )
    titles = unified_search_by_title(title, top_k=top_k)
    if titles is None or titles.empty:
        raise HTTPException(
            status_code=404,
            detail=f'No movie found with title "{title}". Please check for typos or try a different movie.'
        )
    # Ensure it's a DataFrame
    if isinstance(titles, pd.Series):
        titles = titles.to_frame(name="title")  # or give it a proper column name
    return titles.to_dict(orient="records")

@app.get("/recommend/by-genre")
def get_recommendations_by_genre(
    genre: str,
    top_k: int = 10,
    seed: str ='random',
    mode: str ='AND'
):
    """
    Rule-based genre recommendation.
    Provide one or more genres (comma-separated)
    Warns if any input genre does not match known genres.
    """
    if not genre or not genre.strip():
        raise HTTPException(
        status_code=400,
        detail="Genre query cannot be empty."
    )
    genre_list = [g.strip() for g in genre.split(",")] if genre else []
    # Validate partial matches
    unmatched = []
    for g in genre_list:
        if not any(real.lower().startswith(g.lower()) for real in unique_genre):
            unmatched.append(g)

    if unmatched:
        raise HTTPException(
            status_code=400,
            detail=f"The following genre(s) are invalid: {', '.join(unmatched)}"
        )

    recommendations = unified_recommend_by_genres(
        genres=genre_list,
        top_k=top_k,
        seed=seed,
        mode=mode
)
    if isinstance(recommendations, str):
        raise HTTPException(status_code=400, detail=recommendations)

    if recommendations.empty:
        raise HTTPException(
            status_code=404,
            detail="No movies found matching all genres")
    
    return recommendations.to_dict(orient="records")
    