# 🎬 Movie Ratings Analyzer — NumPy + Pandas

A Python-based movie ratings analysis project built using **NumPy, Pandas, Matplotlib, and Streamlit**. This project explores the MovieLens dataset to analyze movie ratings, identify popular movies, study user behavior, compare movie genres, and recommend similar movies using cosine similarity.

The project started as a NumPy-based movie rating analyzer and was later upgraded with Pandas to make data manipulation, analysis, and insights more efficient.

---

## 📌 Project Overview

The Movie Ratings Analyzer works with the MovieLens `ml-latest-small` dataset.

It analyzes movie ratings provided by users and extracts useful insights through numerical computations, data analysis, and visualizations.

### Dataset Information

| Information | Value |
|---|---:|
| Total Movies | 9,742 |
| Total Ratings | 100,836 |
| Total Users | 610 |
| Minimum Rating | 0.5 |
| Maximum Rating | 5.0 |

**Dataset:** [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/)

---

## 🚀 Project Features

### 1. Dataset Exploration

- Load movie and rating datasets using Pandas.
- Explore dataset structure and dimensions.
- Inspect columns, data types, and missing values.
- Identify duplicate records.
- Validate movie IDs and user IDs.
- Check rating values and data consistency.

### 2. Movie Rating Analysis

- Calculate average ratings for individual movies.
- Count the total ratings received by each movie.
- Identify the most-rated movies.
- Find the highest-rated movies.
- Filter movies using a minimum rating-count threshold.
- Sort and compare movie performance.

### 3. User Rating Analysis

- Calculate the number of ratings submitted by each user.
- Identify the most active users.
- Calculate average ratings given by users.
- Explore differences in user rating behavior.

### 4. Genre Analysis

- Extract genres from movie metadata.
- Split multiple genres into individual rows using Pandas.
- Calculate average ratings for genres.
- Calculate total ratings received by each genre.
- Calculate weighted average ratings.

**Weighted Average Formula:**

\[
\text{Weighted Average} =
\frac{\sum(\text{Movie Average} \times \text{Rating Count})}
{\sum(\text{Rating Count})}
\]

This approach accounts for the number of ratings received by each movie rather than treating every movie average equally.

### 5. Movie Recommendation System

The project uses **cosine similarity** to identify movies with similar user-rating patterns.

The recommendation process:

1. Create unique movie and user index mappings.
2. Construct a movie-user rating matrix.
3. Represent each movie as a vector of user ratings.
4. Calculate cosine similarity between movies.
5. Exclude the selected movie from its recommendations.
6. Return the top 5 most similar movies.

**Cosine Similarity Formula:**

\[
\text{Similarity}(A,B) =
\frac{A \cdot B}{\|A\|\|B\|}
\]

The optimized NumPy implementation uses:

- Matrix multiplication (`@`)
- Vectorized calculations
- `np.linalg.norm()`
- `np.divide()` with zero-denominator handling
- `np.argsort()` to identify the most similar movies

This avoids calculating cosine similarity through a separate Python loop for every movie.

**Note:** This is a collaborative-filtering-style recommendation approach based on user-rating vectors. Missing ratings are represented as zero in the matrix.

### 6. Data Visualization with Matplotlib

Three visualizations were added to understand the dataset more clearly.

**Rating Distribution**
- Displays the frequency of each rating.
- Helps understand how users rate movies.

**Top 10 Most-Rated Movies**
- Displays movies with the highest number of ratings.
- Uses a horizontal bar chart for readable movie titles.

**Weighted Average Rating by Genre**
- Compares weighted average ratings across genres.
- Helps identify highly rated and lower-rated genres.

### 7. Streamlit Dashboard

A local interactive dashboard is available through `app.py`.

Current dashboard sections:

- Dataset Overview
- Rating Statistics
- Top 10 Most Rated Movies
- Top 10 Highest Rated Movies
- User Rating Analysis
- Genre Analysis
- Find Similar Movies

The dashboard provides:
- Dataset metrics
- Data tables
- Interactive navigation
- Rating statistics
- Genre-wise weighted average analysis
- Movie selection and similar movie recommendations

The Streamlit dashboard uses its own analysis and display logic in `app.py`, while `main.py` contains the standalone analysis workflow and Matplotlib visualizations.

---

## 🆚 What Changed from the Original NumPy Project?

The original project focused mainly on learning NumPy and implementing movie-rating analysis using arrays and numerical operations.

The updated project combines NumPy with Pandas to make the analysis more practical.

| Area | Original NumPy Version | Updated NumPy + Pandas Version |
|---|---|---|
| Dataset loading | NumPy-based CSV handling | `pd.read_csv()` |
| Data inspection | Array inspection | `head()`, `info()`, `shape`, `describe()` |
| Data cleaning | NumPy operations | Pandas missing-value and duplicate checks |
| Movie statistics | Array-based calculations | `groupby()` and `agg()` |
| Sorting | `np.argsort()` | `sort_values()` |
| Filtering | Boolean masking | Pandas filtering |
| Movie information | Array indexing | DataFrame operations |
| Combining data | Manual alignment | `merge()` |
| Genre analysis | Limited array-based handling | `str.split()` and `explode()` |
| Weighted averages | Numerical calculations | Pandas grouping and aggregation |
| Recommendations | NumPy cosine similarity | Optimized NumPy cosine similarity retained |
| Visualization | Not included initially | Matplotlib charts |
| Frontend | Streamlit dashboard | Expanded Streamlit dashboard with genre analysis |

### Why Keep Both NumPy and Pandas?

Both libraries serve different purposes.

**Pandas** is used for:
- Loading and exploring structured datasets.
- Cleaning and validating data.
- Grouping and aggregating records.
- Joining datasets.
- Extracting insights.

**NumPy** is used for:
- Numerical computation.
- Array manipulation.
- Building the movie-user rating matrix.
- Vectorized mathematical operations.
- Optimized cosine similarity calculations.

The goal was not to replace NumPy with Pandas, but to understand how both libraries work together in a practical data analysis project.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computing and recommendation engine |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Streamlit | Interactive local dashboard |
| MovieLens Dataset | Movie and user rating data |
| Git & GitHub | Version control and project hosting |

---

## 📂 Project Structure

```text
Movie-Ratings-Analyzer_Using_Numpy/
│
├── app.py
│
├── main.py
│
├── README.md
│
├── requirements.txt
│
├── dataset/
│   ├── movies.csv
│   └── ratings.csv
│
└── .gitignore
```

### File Responsibilities

**`main.py`**
- Standalone movie rating analysis.
- NumPy and Pandas operations.
- Movie-user rating matrix.
- Optimized cosine similarity.
- Genre analysis.
- Matplotlib visualizations.

**`app.py`**
- Streamlit frontend.
- Dataset overview and metrics.
- Movie and user analysis.
- Genre analysis.
- Interactive movie recommendations.

**`dataset/`**
- Contains the MovieLens movie and rating CSV files.

**`requirements.txt`**
- Contains the Python dependencies needed to run the project.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project

```bash
cd Movie-Ratings-Analyzer_Using_Numpy
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If required, install the libraries manually:

```bash
pip install numpy pandas matplotlib streamlit
```

### 5. Run the Analysis

```bash
python main.py
```

This executes the analysis and displays the Matplotlib visualizations.

### 6. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 📚 Concepts Learned

This project helped me apply the following concepts practically.

### NumPy
- Arrays and array creation
- Boolean indexing
- Array masking
- Broadcasting
- Vectorization
- Matrix multiplication
- Unique values and indexing
- Sorting and ranking
- Linear algebra
- Vector magnitude
- Cosine similarity
- Handling zero denominators

### Pandas
- Series and DataFrames
- CSV loading
- Dataset inspection
- Missing-value detection
- Duplicate detection
- Data validation
- Filtering and sorting
- `groupby()`
- `agg()`
- `merge()`
- String operations
- `explode()`
- Weighted averages
- Converting Pandas data into NumPy arrays using `to_numpy()`

### Matplotlib
- Creating figures
- Bar charts
- Horizontal bar charts
- Chart titles and labels
- Axis configuration
- Figure sizing
- Layout adjustments

### Streamlit
- Page configuration
- Dashboard layout
- Sidebar navigation
- Select boxes
- Metrics
- DataFrames
- Interactive buttons
- Conditional page rendering

---

## 🎯 Key Learning Outcome

The biggest takeaway from this project was understanding how to combine different Python libraries to solve a practical problem.

Instead of only learning individual functions, I applied NumPy and Pandas together to explore a real dataset, calculate meaningful statistics, build a recommendation engine, and create visual representations of the results.

This project also helped me understand the difference between numerical computing with NumPy and structured data analysis with Pandas.

---

## 🔮 Future Improvements

Some possible improvements for future versions:

- Add movie search and filtering.
- Display movie recommendation explanations.
- Improve the recommendation system with additional techniques.
- Add interactive Matplotlib charts to the Streamlit dashboard.
- Improve dashboard design and user experience.
- Add content-based recommendations using movie genres.
- Deploy the dashboard publicly.

---

## 👨‍💻 Author

**Syam Prasad**

B.Tech Computer Science and Engineering

- GitHub: [Syamprasad1212](https://github.com/Syamprasad1212)
- LinkedIn: [Syam Prasad](https://www.linkedin.com/in/syam-vadarevula-2a959a303/)

---

*Built as a hands-on learning project to understand NumPy, Pandas, data analysis, recommendation systems, and Python visualization.*