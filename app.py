
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Ratings Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# 2. Load Dataset
# -----------------------------
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

ratings_values = ratings["rating"].to_numpy()
rated_movie_ids = ratings["movieId"].to_numpy()
user_ids = ratings["userId"].to_numpy()

unique_movie_ids, rating_counts = np.unique(
    rated_movie_ids, return_counts=True
)

unique_user_ids, user_rating_counts = np.unique(
    user_ids, return_counts=True
)

# -----------------------------
# 3. Calculate Average Rating Per Movie
# -----------------------------
average_ratings = []

for movie_id in unique_movie_ids:
    movie_ratings = ratings_values[rated_movie_ids == movie_id]
    average_ratings.append(np.mean(movie_ratings))

average_ratings = np.array(average_ratings)

# -----------------------------
# 4. Dashboard Title
# -----------------------------
st.title("🎬 Movie Ratings Analyzer")
st.write("Explore movie ratings, popular movies and similar movies.")

# -----------------------------
# 5. Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")

section = st.sidebar.selectbox(
    "Choose an Analysis",
    [
        "Dataset Overview",
        "Rating Statistics",
        "Top 10 Most Rated Movies",
        "Top 10 Highest Rated Movies",
        "User Rating Analysis",
        "Genre Analysis",
        "Find Similar Movies"
    ]
)

# -----------------------------
# 6. Dataset Overview
# -----------------------------
if section == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(movies))
    col2.metric("Total Ratings", len(ratings))
    col3.metric("Total Users", len(unique_user_ids))

    st.subheader("Dataset Preview")
    st.dataframe(movies.head(10), use_container_width=True)

# -----------------------------
# 7. Rating Statistics
# -----------------------------
elif section == "Rating Statistics":

    st.header("Rating Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Rating", f"{np.mean(ratings_values):.2f}")
    col2.metric("Highest Rating", np.max(ratings_values))
    col3.metric("Lowest Rating", np.min(ratings_values))

    unique_ratings, counts = np.unique(
        ratings_values, return_counts=True
    )

    rating_data = pd.DataFrame({
        "Rating": unique_ratings,
        "Number of Ratings": counts
    })

    st.subheader("Rating Distribution")
    st.bar_chart(rating_data.set_index("Rating"))

    st.subheader("Rating Distribution")

    rating_counts = ratings["rating"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        rating_counts.index,
        rating_counts.values
    )

    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    ax.set_xticks(rating_counts.index)

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# 8. Top 10 Most Rated Movies
# -----------------------------
elif section == "Top 10 Most Rated Movies":

    st.header("🏆 Top 10 Most Rated Movies")

    sorted_indices = np.argsort(rating_counts)[::-1][:10]

    top_movie_ids = unique_movie_ids[sorted_indices]
    top_counts = rating_counts[sorted_indices]

    result = movies[movies["movieId"].isin(top_movie_ids)].copy()

    result = result.set_index("movieId").loc[top_movie_ids].reset_index()

    result["Rating Count"] = top_counts

    st.dataframe(
        result[["title", "Rating Count"]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Most Popular Movies")
    st.bar_chart(result.set_index("title")["Rating Count"])

    st.subheader("Top 10 Most Rated Movies (Matplotlib)")

    chart_data = result.sort_values(
        "Rating Count",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        chart_data["title"],
        chart_data["Rating Count"]
    )

    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Movie")
    ax.set_title("Top 10 Most Rated Movies")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# 9. Top 10 Highest Rated Movies
# -----------------------------
elif section == "Top 10 Highest Rated Movies":

    st.header("⭐ Top 10 Highest Rated Movies")

    popular_mask = rating_counts >= 50

    popular_movie_ids = unique_movie_ids[popular_mask]
    popular_averages = average_ratings[popular_mask]
    popular_counts = rating_counts[popular_mask]

    sorted_indices = np.argsort(popular_averages)[::-1][:10]

    top_movie_ids = popular_movie_ids[sorted_indices]
    top_averages = popular_averages[sorted_indices]
    top_counts = popular_counts[sorted_indices]

    result = movies[movies["movieId"].isin(top_movie_ids)].copy()

    result = result.set_index("movieId").loc[top_movie_ids].reset_index()

    result["Average Rating"] = np.round(top_averages, 2)
    result["Rating Count"] = top_counts

    st.caption("Showing movies with at least 50 ratings.")

    st.dataframe(
        result[["title", "Average Rating", "Rating Count"]],
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# 10. User Rating Analysis
# -----------------------------
elif section == "User Rating Analysis":

    st.header("👥 User Rating Analysis")

    col1, col2 = st.columns(2)

    col1.metric("Total Users", len(unique_user_ids))
    col2.metric("Total Ratings", len(ratings))

    sorted_indices = np.argsort(user_rating_counts)[::-1][:10]

    top_users = unique_user_ids[sorted_indices]
    top_counts = user_rating_counts[sorted_indices]

    st.subheader("Top 10 Most Active Users")

    st.dataframe(
        pd.DataFrame({
            "User ID": top_users,
            "Ratings Given": top_counts
        }),
        use_container_width=True,
        hide_index=True
    )

    average_user_ratings = []

    for user_id in unique_user_ids:
        user_ratings = ratings_values[user_ids == user_id]
        average_user_ratings.append(np.mean(user_ratings))

    average_user_ratings = np.array(average_user_ratings)

    st.subheader("Average Ratings by Users")

    st.dataframe(
        pd.DataFrame({
            "User ID": unique_user_ids[:10],
            "Average Rating": np.round(average_user_ratings[:10], 2)
        }),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# 11. Genre Analysis
# -----------------------------

elif section == "Genre Analysis":

    st.header("🎭 Genre Analysis")

    # Calculate movie-wise rating statistics
    movie_stats = ratings.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    # Merge movie information
    genre_data = movies.merge(movie_stats, on="movieId")

    # Split multiple genres into separate rows
    genre_data["genres"] = genre_data["genres"].str.split("|")

    genre_data = genre_data.explode("genres")

    # Calculate weighted average rating per genre
    weighted_genre_stats = genre_data.groupby("genres").apply(
        lambda group: (
            (group["average_rating"] * group["rating_count"]).sum()
            / group["rating_count"].sum()
        ),
        include_groups=False
    ).reset_index(name="weighted_average")

    # Display results
    weighted_genre_stats = weighted_genre_stats.sort_values(
        "weighted_average",
        ascending=False
    )

    st.subheader("Weighted Average Rating by Genre")

    st.dataframe(
        weighted_genre_stats.round(2),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------
# 11. Find Similar Movies
# -----------------------------
elif section == "Find Similar Movies":

    st.header("🎥 Find Similar Movies")

    movie_titles = movies["title"].to_numpy()

    selected_title = st.selectbox(
        "Choose a movie",
        movie_titles
    )

    if st.button("Find Similar Movies"):

        target_movie_id = movies.loc[
            movies["title"] == selected_title, "movieId"
        ].iloc[0]

        # Build movie-user rating matrix
        unique_users = np.unique(user_ids)

        rating_matrix = np.zeros(
            (len(unique_movie_ids), len(unique_users))
        )

        movie_index = {
            movie_id: index
            for index, movie_id in enumerate(unique_movie_ids)
        }

        user_index = {
            user_id: index
            for index, user_id in enumerate(unique_users)
        }

        for user, movie, rating in zip(
            user_ids, rated_movie_ids, ratings_values
        ):
            rating_matrix[
                movie_index[movie], user_index[user]
            ] = rating

        # Calculate cosine similarity
        target_index = movie_index.get(target_movie_id)

        target_vector = rating_matrix[target_index]

        dot_products = rating_matrix @ target_vector

        magnitudes = np.linalg.norm(rating_matrix, axis=1)
        target_magnitude = np.linalg.norm(target_vector)

        denominators = magnitudes * target_magnitude

        similarities = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products),
            where=denominators != 0
        )

        # Exclude selected movie
        similarities[target_index] = -1

        # Find top 5 similar movies
        top_indices = np.argsort(similarities)[::-1][:5]

        similar_ids = unique_movie_ids[top_indices]
        similar_scores = similarities[top_indices]

        result = movies[
            movies["movieId"].isin(similar_ids)
        ].copy()

        result = result.set_index("movieId").loc[similar_ids].reset_index()

        result["Similarity Score"] = np.round(similar_scores, 4)

        st.subheader(f"Movies Similar to {selected_title}")

        st.dataframe(
            result[["title", "Similarity Score"]],
            use_container_width=True,
            hide_index=True
        )

# -----------------------------
# 12. Footer
# -----------------------------
st.sidebar.divider()
st.sidebar.caption("Movie Ratings Analyzer | Built with NumPy, Pandas, Matplotlib, & Streamlit")
