
import numpy as np
import pandas as pd


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
# 4. HIGHEST-RATED MOVIES ANALYSIS
# ==================================================

# Calculate average rating for every movie
average_ratings = []

for movie_id in unique_movie_ids:

    # Select ratings belonging to the current movie
    movie_ratings = ratings_values[rated_movie_ids == movie_id]

    avg = np.mean(movie_ratings)
    average_ratings.append(avg)

average_ratings = np.array(average_ratings)

# Consider only movies with at least 50 ratings
popular_mask = rating_counts >= 50

popular_movie_ids = unique_movie_ids[popular_mask]
popular_average_ratings = average_ratings[popular_mask]
popular_rating_counts = rating_counts[popular_mask]

print("\nMovies with at least 50 ratings:", len(popular_movie_ids))

# Sort by average rating in descending order
sorted_indices = np.argsort(popular_average_ratings)[::-1]

top_10_indices = sorted_indices[:10]

top_movie_ids = popular_movie_ids[top_10_indices]
top_average_ratings = popular_average_ratings[top_10_indices]
top_rating_counts = popular_rating_counts[top_10_indices]

print("\n========== TOP 10 HIGHEST-RATED MOVIES ==========")

for movie_id, avg, count in zip(
    top_movie_ids,
    top_average_ratings,
    top_rating_counts
):
    title = movie_title_by_id[movie_id]
    print(f"{title} | Average: {avg:.2f} | Ratings: {count}")


# ==================================================
# 5. USER RATING ANALYSIS
# ==================================================

# Find unique users and count ratings given by each user
unique_user_ids, user_rating_counts = np.unique(
    user_ids,
    return_counts=True
)

print("\n========== USER STATISTICS ==========")

print("Total Unique Users:", len(unique_user_ids))

# Find the 10 most active users
sorted_indices = np.argsort(user_rating_counts)[::-1]

top_10_user_indices = sorted_indices[:10]

top_users = unique_user_ids[top_10_user_indices]
top_user_counts = user_rating_counts[top_10_user_indices]

print("\n========== TOP 10 MOST ACTIVE USERS ==========")

for user, count in zip(top_users, top_user_counts):
    print(f"User ID: {user} | Ratings: {count}")

# Calculate average rating given by each user
average_user_ratings = []

for user_id in unique_user_ids:

    # Select all ratings given by the current user
    user_ratings = ratings_values[user_ids == user_id]

    avg = np.mean(user_ratings)
    average_user_ratings.append(avg)

average_user_ratings = np.array(average_user_ratings)

print("\n========== AVERAGE RATINGS BY USERS ==========")

for user_id, avg in zip(
    unique_user_ids[:10],
    average_user_ratings[:10]
):
    print(f"User ID: {user_id} | Average Rating: {avg:.2f}")


# ==================================================
# 6. MOVIE SIMILARITY USING COSINE SIMILARITY
# ==================================================

# Create a matrix where:
# Rows = Movies
# Columns = Users
# Values = Ratings
# Zero = No rating given

rating_matrix = np.zeros(
    (len(unique_movie_ids), len(unique_user_ids))
)

# Map movie IDs and user IDs to their matrix positions
movie_index = {
    movie_id: index
    for index, movie_id in enumerate(unique_movie_ids)
}

user_index = {
    user_id: index
    for index, user_id in enumerate(unique_user_ids)
}

# Fill the rating matrix using the original ratings
for user, movie, rating in zip(
    user_ids,
    rated_movie_ids,
    ratings_values
):
    m_index = movie_index[movie]
    u_index = user_index[user]

    rating_matrix[m_index, u_index] = rating

print("\n========== MOVIE RATING MATRIX ==========")
print("Rating Matrix Shape:", rating_matrix.shape)


# ==================================================
# 7. FIND TOP 5 SIMILAR MOVIES
# ==================================================

# Select the target movie (Toy Story)
target_movie_id = 1

target_index = movie_index[target_movie_id]
target_vector = rating_matrix[target_index]

# Calculate target movie magnitude once
magnitude_target = np.linalg.norm(target_vector)

similarities = []

# Compare target movie with every other movie
for i in range(len(unique_movie_ids)):

    # Skip the target movie itself
    if i == target_index:
        continue

    other_vector = rating_matrix[i]

    dot_product = np.dot(target_vector, other_vector)
    magnitude_other = np.linalg.norm(other_vector)

    # Calculate cosine similarity
    if magnitude_target != 0 and magnitude_other != 0:
        similarity = dot_product / (
            magnitude_target * magnitude_other
        )
    else:
        similarity = 0

    similarities.append((unique_movie_ids[i], similarity))

# Sort movies by similarity score
similarities.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\n========== TOP 5 SIMILAR MOVIES ==========")
print("Reference Movie:", movie_title_by_id[target_movie_id])

for movie_id, score in similarities[:5]:

    title = movie_title_by_id[movie_id]

    print(f"{title} | Similarity: {score:.4f}")


# ==================================================
# END OF MOVIE RATINGS ANALYZER
# ==================================================

print("\n========== ANALYSIS COMPLETED ==========")
