# Hybrid Movie Recommendation System

**A production-ready hybrid recommender** that merges **Collaborative Filtering (CF)**, **Content-Based Filtering (CBF)**, and **Transfer Learning** to deliver personalized movie suggestions — even for films without user ratings.

This project tackles the **cold-start problem** by mapping knowledge between two distinct datasets (**MovieLens** and **TMDb**), allowing recommendations to work across both rating-rich and rating-scarce movies.

---

## Problem Statement
Streaming platforms like Netflix, Amazon Prime, and Disney+ face two key challenges:
1. **Cold Starts** – new movies with no ratings cannot be recommended using Collaborative Filtering alone.
2. **Data Fragmentation** – information about the same movie is often spread across different datasets.

This project solves both problems by:
- Using **CF** where ratings exist.
- Using **CBF** where ratings are missing.
- **Bridging datasets** via intersection mapping for improved coverage and accuracy.

---

## System Architecture

```plaintext
      ┌─────────────────────┐
      │ Dataset 1: MovieLens│───► Collaborative Filtering
      └─────────────────────┘
                 ▲
                 │
Intersection Mapping (3,534 movies)
                 │
                 ▼
      ┌─────────────────────┐
      │ Dataset 2: TMDb     │───► Content-Based Filtering
      └─────────────────────┘

               Hybrid Scoring
           (Weighting + Merging)
                    │
                    ▼
          Unified Ranked Recommendations
```
## Key Features
- **Collaborative Filtering (CF)** – Learns from user–item rating patterns (Dataset 1).
- **Content-Based Filtering (CBF)** – Uses cosine similarity over genres and tags (Dataset 2).
- **Transfer Learning** – Bridges datasets through a shared set of 3,534 movies.
- **Cold Start Handling** – Generates recommendations for new movies without ratings.
- **Weighted Hybrid Strategy**:

    - Prefers Dataset 2 similarity when available.

    - Scales Dataset 1-only recommendations by a weight of 0.85.

- **FastAPI Integration** – Deployable REST API with JSON responses.
- **Extensible Design** – Can be adapted to books, music, or other domains.

---

## Performance Metrics

| Metric        | Value   | Dataset           |
|---------------|---------|-------------------|
| RMSE          | 0.8686  | CF on Dataset 1   |
| Precision@10  | 0.6432  | CF on Dataset 1   |
| Recall@10     | 0.6794  | CF on Dataset 1   |

The collaborative filtering model achieved an RMSE of 0.8672 (ratings on a 1–5 scale), indicating high predictive accuracy. Precision@10 (0.6432) and Recall@10 (0.6794) reflect a balanced ability to return relevant recommendations while covering a wide range of relevant items. The system performs well even for cold-start users by leveraging content-based signals, and can be extended to track new user activity (e.g., ratings or watch history) to further refine and personalize recommendations over time.

---

## Project Structure
```
hybrid_recommender/
│
├── data/
│   ├── raw/
│   │   ├── dataset1/
│   │   ├── dataset2/
│   ├── processed/
│   │   ├── dataset1/
│   │   ├── dataset2/
│   ├── intersection/
│
├── notebooks/
│   ├── phase1_eda.ipynb
│   ├── phase2_eda.ipynb
│   ├── intersection_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── dataset1_preprocess.py
│   │   ├── dataset2_preprocess.py
│   ├── recommenders/
│   │   ├── collaborative.py
│   │   ├── content_based.py
│   │   ├── hybrid.py
│   ├── utils/
│   │   ├── evaluation.py    # metrics functions
│   ├── artifacts/           # pickle/model files
│
├── main.py                   # FastAPI entrypoint
├── app.py                  # Streamlit entrypoint
├── requirements.txt
└── README.md

```

---

## Datasets
- **Dataset 1** – [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (explicit ratings)

- **Dataset 2** – [TMDb](https://www.themoviedb.org/) (rich metadata: genres, tags, cast, crew)

- **Intersection** – 3,534 movies in both datasets for transfer learning.

```⚠️ TMDb data is used under its terms of use.```

---

## How It Works
1. Preprocessing
    - Parse and clean metadata (genres, tags, etc.).
    - Build cosine similarity matrices for genres and tags.

2. Recommendation Generation
    - CF from Dataset 1 for known users.
    - CBF from Dataset 2 for cold starts.

3. Hybrid Merging
    - Merge CF + CBF scores.
    - Apply intersection mapping and weighting rules.
    - Deduplicate and rank.

4. API Serving

    - Endpoints for:
        - Recommend by user ID (Hybrid recommender)
        - Recommend by movie title
        - Recommend by movie genre(s)
        - Search movies by title
    - JSON responses with titles and scores.

---

## Installation
```
git clone https://github.com/ChibuikeNwankwo/hybrid-movie-recommender.git
cd hybrid-movie-recommender
pip install -r requirements.txt
```

---

## Usage
- Start the FastAPI server:
```
uvicorn main:app --reload 
```
API Endpoints:
```
GET /recommend/hybrid/{user_id}?top_k=10
GET /recommend/by-title/{movie_title}?top_k=10
GET /recommend/by-genre/{genre}?top_k=10
GET /search/by-title/{query}?top_k=10
```
Example JSON Output:
```
[
  {"title": "Inception", "score": 0.9234},
  {"title": "Interstellar", "score": 0.9152}
]
```
---

## Demo
- **Loom Video:** [Loom Video](https://www.loom.com/share/01d44185498e4dc288f2e23894f27abb?sid=f5d982bc-5bfb-48e2-8187-53e428bb3dee)
- **Streamlit App:** Run locally: 
```
streamlit run app.py
```
- **GitHub Repo:** [Github](https://github.com/ChibuikeNwankwo/hybrid-movie-recommender)

---

## Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Surprise)
- **FastAPI**
- **Streamlit** (optional UI)
- **Datasets:** MovieLens, TMDb

## License
- **MIT License** – feel free to use and modify.
---
**Author:** Chibuike Nwankwo – [LinkedIn](https://www.linkedin.com/in/chibuike-nwankwo55 )| [GitHub Portfolio](https://github.com/ChibuikeNwankwo)













