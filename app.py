import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

page = st.sidebar.selectbox(
    "Choose an option:",
    ["Home","Recommend for User", "Similar Movies by Title", "Search by Title", "Similar Movies by Genre"]
)

if page == "Home":
    st.title("🎬 Welcome to the Movie Recommender System")
    st.markdown("""
    This app demonstrates a **hybrid recommender system** built with:
    - 🔹 FastAPI (Backend API)
    - 🔹 Streamlit (Frontend UI)
    - 🔹 Hybrid of collaborative & content-based filtering  

    ### What you can do:
    - 👤 Get recommendations for a **specific user**
    - 🎥 Find **similar movies by title**
    - 🏷️ Explore **movies by genre**
    - 🔎 Search for a movie by title  

    ---
    👈 Use the sidebar to explore the app.
    """)


# 1. Hybrid Recommender
if page == "Recommend for User":
    user_id = st.number_input("Enter User ID:", min_value=0, step=1, value=0)
    top_k = st.slider("Number of recommendations:", 5, 20, 10)
    if user_id and user_id >= 1:
        if st.button("Get Recommendations"):
            with st.spinner("Fetching Recommendations..."):
                response = requests.get(
                    f"{API_URL}/recommend/hybrid",
                    params={"user_id": user_id, "top_k": top_k}
                )
            if response.ok:
                st.write("### Recommended Movies:")
                for rec in response.json():
                    st.write(f"- {rec['title']}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
    else:
        st.info("🔒 Enter a valid User ID to enable search.")


# 2. Similar Movies by Title
elif page == "Similar Movies by Title":
    title = st.text_input("Enter a movie title:")
    top_k = st.slider("Number of similar movies:", 5, 20, 10)
    
    if st.button("Find Similar Movies"):
        if title.strip():  # ✅ prevent empty input
            with st.spinner('Finding Movies...'):
                response = requests.get(
                    f"{API_URL}/recommend/by-title",
                    params={"title": title, "top_k": top_k}
                )
                if response.ok:
                    st.write("### Similar Movies:")
                    for rec in response.json():
                        st.write(f"- {rec['title']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
        else:
            st.warning("⚠️ Please enter a movie title before searching.")


# 3. Search by Title
elif page == "Search by Title":
    title = st.text_input("Enter a movie title to search:")
    top_k = st.slider("Number of search results:", 5, 20, 10)
    if st.button("Search"):
        if not title.strip():
            st.warning("⚠️ Please enter a movie title before searching.")
        else:
            with st.spinner('Searching Titles...'):
                response = requests.get(
                f"{API_URL}/search/by-title",
                params={"title": title, "top_k": top_k}
            )
                if response.ok:
                    st.write("### Search Results:")
                    for rec in response.json():
                        st.write(f"- {rec['title']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")


# 4. Similar Movies by Genre
elif page == "Similar Movies by Genre":
    genre = st.text_input("Enter genre(s), comma-separated (e.g., Action, Comedy):")
    top_k = st.slider("Number of movies:", 5, 20, 10)
    if st.button("Find by Genre"):
        if not genre.strip():
            st.warning("⚠️ Please enter at least one genre before searching.")
        else:
            with st.spinner('Fetching recommendations...'):
                response = requests.get(
                f"{API_URL}/recommend/by-genre",
                params={"genre": genre, "top_k": top_k}
            )
                if response.ok:
                    st.write("### Movies by Genre:")
                for rec in response.json():
                    st.write(f"- {rec['title']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
