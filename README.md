# 🎬 Movie Ratings Analyzer Using NumPy

A Python-based movie ratings analysis project built using **NumPy, Pandas, and Streamlit**. This project analyzes the MovieLens dataset to explore movie popularity, rating patterns, user activity, and movie similarities through an interactive web dashboard.

## 📌 Project Overview

The Movie Ratings Analyzer uses NumPy for numerical computations and Pandas for dataset handling. Streamlit provides a simple interactive dashboard where users can explore different movie rating analyses.

## 🚀 Features

- **Dataset Overview:** View the total number of movies, ratings, and users.
- **Rating Statistics:** Calculate average, highest, and lowest ratings.
- **Top 10 Most Rated Movies:** Identify movies with the highest number of ratings.
- **Top 10 Highest Rated Movies:** Find highly rated movies with at least 50 ratings.
- **User Rating Analysis:** Explore active users and their average ratings.
- **Find Similar Movies:** Identify similar movies using cosine similarity.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Streamlit
- MovieLens Dataset

## 📂 Project Structure

```text
Movie-Ratings-Analyzer/
│
├── dataset/
│   ├── movies.csv
│   └── ratings.csv
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Syamprasad1212/Movie-Ratings-Analyzer_Using_Numpy.git
```

### 2. Navigate to the project directory

```bash
cd Movie-Ratings-Analyzer_Using_Numpy
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open in your browser

```text
http://localhost:8501
```

## 📊 Dataset

This project uses the **MovieLens Latest Small Dataset** provided by GroupLens Research.

Dataset source: https://grouplens.org/datasets/movielens/latest/

The dataset contains movie information and user ratings.

## 🧠 Concepts Practiced

- NumPy arrays and array operations
- Array indexing and Boolean masking
- Statistical calculations
- Unique values and frequency counting
- Sorting and filtering arrays
- Dot product and vector magnitude
- Cosine similarity
- Pandas DataFrame operations
- Interactive dashboards using Streamlit

## 🔮 Future Improvements

- Add movie posters and descriptions.
- Improve the dashboard UI.
- Add genre-based filtering.
- Explore more advanced recommendation techniques.

## 👨‍💻 Author

**Syam Prasad**

- GitHub: [Syamprasad1212](https://github.com/Syamprasad1212)

---

⭐ If you find this project useful, consider giving it a star!
