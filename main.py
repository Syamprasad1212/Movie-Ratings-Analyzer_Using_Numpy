
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. LOAD DATASET AND CONVERT COLUMNS INTO NUMPY ARRAYS
# ==================================================

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Convert Pandas columns into NumPy arrays
movie_ids = movies["movieId"].to_numpy()
movie_titles = movies["title"].to_numpy()

user_ids = ratings["userId"].to_numpy()
rated_movie_ids = ratings["movieId"].to_numpy()
ratings_values = ratings["rating"].to_numpy()

# Create a dictionary to quickly find movie titles using movie IDs
movie_title_by_id = dict(zip(movie_ids, movie_titles))


# ==================================================
# 2. BASIC DATASET AND RATING STATISTICS
# ==================================================

print("\n========== DATASET OVERVIEW ==========")

print("Movies Dataset Shape:", movies.shape)
print("Ratings Dataset Shape:", ratings.shape)

print("\n========== RATING ARRAY INFORMATION ==========")

print("Array Shape:", ratings_values.shape)
print("Array Dimensions:", ratings_values.ndim)
print("Data Type:", ratings_values.dtype)
print("Total Elements:", ratings_values.size)

# Calculate basic rating statistics
average_rating = np.mean(ratings_values)
highest_rating = np.max(ratings_values)
lowest_rating = np.min(ratings_values)
total_ratings = np.size(ratings_values)

print("\nAverage Rating:", average_rating)
print("Highest Rating:", highest_rating)
print("Lowest Rating:", lowest_rating)
print("Total Ratings:", total_ratings)

# Find unique rating values and their frequencies
unique_ratings = np.unique(ratings_values)

print("\n========== RATING FREQUENCY ==========")

for rating in unique_ratings:
    count = np.sum(ratings_values == rating)
    print(f"{rating}: {count}")


# ==================================================
# 3. MOST-RATED MOVIES ANALYSIS
# ==================================================

# Find unique movie IDs and count their ratings
unique_movie_ids, rating_counts = np.unique(
    rated_movie_ids,
    return_counts=True
)

print("\nTotal Unique Movies Rated:", len(unique_movie_ids))

# Sort movies by rating count in descending order
sorted_indices = np.argsort(rating_counts)[::-1]

# Select top 10 most-rated movies
top_10_indices = sorted_indices[:10]

top_10_movie_ids = unique_movie_ids[top_10_indices]
top_10_counts = rating_counts[top_10_indices]

print("\n========== TOP 10 MOST-RATED MOVIES ==========")

for movie_id, count in zip(top_10_movie_ids, top_10_counts):
    title = movie_title_by_id[movie_id]
    print(f"{title}: {count}")


# ==================================================
# 4. HIGHEST-RATED MOVIES ANALYSIS USING PANDAS
# ==================================================

# Group ratings by movie ID and calculate statistics
movie_stats = ratings.groupby("movieId")["rating"].agg(
    average_rating="mean",
    rating_count="count"
)

# Consider only movies with at least 50 ratings
popular_movies = movie_stats[
    movie_stats["rating_count"] >= 50
]

print("\nMovies with at least 50 ratings:", len(popular_movies))

# Sort by average rating in descending order
top_movies = popular_movies.sort_values(
    by="average_rating",
    ascending=False
).head(10)

print("\n========== TOP 10 HIGHEST-RATED MOVIES ==========")

for movie_id, row in top_movies.iterrows():
    title = movie_title_by_id[movie_id]

    print(
        f"{title} | Average: {row['average_rating']:.2f} "
        f"| Ratings: {row['rating_count']}"
    )

# Merge movie details with rating statistics
movie_analysis = movies.merge(
    movie_stats,
    on="movieId",
    how="inner"
)

print("\n========== COMBINED MOVIE ANALYSIS ==========")
print(movie_analysis.head())


# ==================================================
# 4.1 ADVANCED MOVIE INSIGHTS USING PANDAS
# ==================================================

# Top 10 most-rated movies
most_rated = movie_analysis.sort_values(
    by="rating_count",
    ascending=False
).head(10)

print("\n========== MOST POPULAR MOVIES ==========")

print(
    most_rated[["title", "rating_count"]].to_string(index=False)
)


# Top 10 highest-rated movies (minimum 50 ratings)
highest_rated = movie_analysis[
    movie_analysis["rating_count"] >= 50
].sort_values(
    by="average_rating",
    ascending=False
).head(10)

print("\n========== HIGHEST-RATED MOVIES ==========")

print(
    highest_rated[
        ["title", "average_rating", "rating_count"]
    ].to_string(index=False)
)


# Find movies with average rating below 2.5
low_rated = movie_analysis[
    movie_analysis["average_rating"] < 2.5
].sort_values(
    by="average_rating"
)

print("\n========== LOWEST-RATED MOVIES ==========")

print(
    low_rated[
        ["title", "average_rating", "rating_count"]
    ].head(10).to_string(index=False)
)


# ==================================================
# 5. USER RATING ANALYSIS USING PANDAS
# ==================================================

# Group ratings by user ID and calculate statistics
user_stats = ratings.groupby("userId")["rating"].agg(
    average_rating="mean",
    rating_count="count"
)

print("\n========== USER STATISTICS ==========")

print("Total Unique Users:", len(user_stats))

# Find the 10 most active users
top_active_users = user_stats.sort_values(
    by="rating_count",
    ascending=False
).head(10)

print("\n========== TOP 10 MOST ACTIVE USERS ==========")

for user_id, row in top_active_users.iterrows():
    print(
        f"User ID: {user_id} | Ratings: {int(row['rating_count'])}"
    )

# Find users with the highest average ratings
top_average_users = user_stats.sort_values(
    by="average_rating",
    ascending=False
).head(10)

print("\n========== TOP 10 USERS BY AVERAGE RATING ==========")

for user_id, row in top_average_users.iterrows():
    print(
        f"User ID: {user_id} | "
        f"Average Rating: {row['average_rating']:.2f}"
    )


# Convert user statistics index into NumPy arrays
unique_user_ids = user_stats.index.to_numpy()
user_rating_counts = user_stats["rating_count"].to_numpy()

# ==================================================
# NUMPY MOVIE-USER RATING MATRIX
# ==================================================

# Create index mappings
movie_index = {
    movie_id: index
    for index, movie_id in enumerate(unique_movie_ids)
}

user_index = {
    user_id: index
    for index, user_id in enumerate(unique_user_ids)
}

# Create rating matrix
rating_matrix = np.zeros(
    (len(unique_movie_ids), len(unique_user_ids))
)

# Fill matrix with ratings
for user_id, movie_id, rating in zip(
    user_ids, rated_movie_ids, ratings_values
):
    movie_position = movie_index[movie_id]
    user_position = user_index[user_id]

    rating_matrix[movie_position, user_position] = rating

print("\nRating Matrix Shape:", rating_matrix.shape)

# ==================================================
# OPTIMIZED COSINE SIMILARITY
# ==================================================

# Toy Story (1995)
reference_movie_id = 1

target_index = np.where(
    unique_movie_ids == reference_movie_id
)[0][0]

target_vector = rating_matrix[target_index]

# Calculate dot products for all movies at once
dot_products = rating_matrix @ target_vector

# Calculate magnitude of every movie vector
movie_magnitudes = np.linalg.norm(
    rating_matrix,
    axis=1
)

target_magnitude = np.linalg.norm(target_vector)

# Calculate cosine similarity safely
denominator = target_magnitude * movie_magnitudes

similarities = np.divide(
    dot_products,
    denominator,
    out=np.zeros_like(dot_products, dtype=float),
    where=denominator != 0
)

# Exclude Toy Story itself
similarities[target_index] = -1

# Get top 5 similar movies
top_indices = np.argsort(similarities)[::-1][:5]

print("\n========== TOP 5 SIMILAR MOVIES ==========")
print("Reference Movie:", movie_title_by_id[reference_movie_id])

for index in top_indices:
    movie_id = unique_movie_ids[index]
    title = movie_title_by_id[movie_id]
    score = similarities[index]

    print(f"{title} | Similarity: {score:.4f}")

# ==================================================
# 6. GENRE-BASED ANALYSIS
# ==================================================

genre_analysis = movie_analysis.assign(
    genres=movie_analysis["genres"].str.split("|")
).explode("genres")

print("\n========== GENRE DATA ==========")
print(genre_analysis[["title", "genres"]].head(10))


# ==================================================
# GENRE-WISE STATISTICS
# ==================================================

genre_stats = genre_analysis.groupby("genres").agg(
    average_rating=("average_rating", "mean"),
    total_movies=("movieId", "nunique"),
    total_ratings=("rating_count", "sum")
)

# Sort genres by average rating
top_genres = genre_stats.sort_values(
    by="average_rating",
    ascending=False
)

print("\n========== GENRE-WISE AVERAGE RATINGS ==========")
print(top_genres.round(2))

# Find most popular genres by movie count
popular_genres = genre_stats.sort_values(
    by="total_movies",
    ascending=False
)

print("\n========== MOST COMMON GENRES ==========")
print(popular_genres[["total_movies"]].head(10))



# ==================================================
# WEIGHTED GENRE RATING ANALYSIS
# ==================================================

# Calculate rating points for each movie-genre row
genre_analysis = genre_analysis.copy()

genre_analysis["rating_points"] = (
    genre_analysis["average_rating"] *
    genre_analysis["rating_count"]
)

# Aggregate rating points and rating counts
weighted_genre_stats = genre_analysis.groupby("genres").agg(
    total_rating_points=("rating_points", "sum"),
    total_ratings=("rating_count", "sum")
)

# Calculate weighted average
weighted_genre_stats["weighted_average"] = (
    weighted_genre_stats["total_rating_points"] /
    weighted_genre_stats["total_ratings"]
)

weighted_genre_stats = weighted_genre_stats.sort_values(
    by="weighted_average",
    ascending=False
)

print("\n========== WEIGHTED GENRE RATINGS ==========")

print(
    weighted_genre_stats[
        ["weighted_average", "total_ratings"]
    ].round(2)
)


# ==================================================
# VISUALIZATION 1: RATING DISTRIBUTION
# ==================================================

rating_counts = ratings["rating"].value_counts().sort_index()

plt.figure(figsize=(10, 5))

plt.bar(rating_counts.index, rating_counts.values)

plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.xticks(np.arange(0.5, 5.1, 0.5))

plt.tight_layout()
plt.show()

# ==================================================
# VISUALIZATION 2: TOP 10 MOST-RATED MOVIES
# ==================================================

top_10_movies = most_rated.head(10).sort_values("rating_count")

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_movies["title"],
    top_10_movies["rating_count"]
)

plt.title("Top 10 Most-Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie")

plt.tight_layout()
plt.show()

# ==================================================
# VISUALIZATION 3: GENRE RATINGS
# ==================================================

genre_plot = weighted_genre_stats.sort_values("weighted_average")

plt.figure(figsize=(10, 8))

plt.barh(
    genre_plot.index,
    genre_plot["weighted_average"]
)

plt.title("Weighted Average Rating by Genre")
plt.xlabel("Weighted Average Rating")
plt.ylabel("Genre")

plt.xlim(0, 5)

plt.tight_layout()
plt.show()