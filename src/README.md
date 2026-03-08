# 🎬 Movie Recommender System - Jupyter Notebook

## Content-Based Movie Recommendation Engine using TMDB Dataset

This notebook implements a content-based movie recommendation system using the TMDB 5000 Movie Dataset. The system recommends movies similar to a user's selected movie based on features like overview, genres, keywords, cast, and crew.

Future Scope - [Movie Knowledge Graph](https://github.com/DuarteDizz/movie-kg-explorer)

---

## 📋 Table of Contents
1. [Importing Libraries](#importing-libraries)
2. [Loading the Dataset](#loading-the-dataset)
3. [Data Exploration & Merging](#data-exploration--merging)
4. [Feature Selection](#feature-selection)
5. [Handling Missing Data](#handling-missing-data)
6. [Data Preprocessing (JSON Parsing)](#data-preprocessing-json-parsing)
7. [Feature Engineering](#feature-engineering)
8. [Text Vectorization (Bag of Words)](#text-vectorization-bag-of-words)
9. [Similarity Calculation](#similarity-calculation)
10. [Recommendation Function](#recommendation-function)
11. [Testing the Recommender](#testing-the-recommender)

---

## 1. Importing Libraries

```python
import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O
import ast           # for safely evaluating strings containing Python literals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

**Explanation:**
- **NumPy**: Provides support for large, multi-dimensional arrays and mathematical functions
- **Pandas**: Used for data manipulation and analysis with DataFrame structures
- **ast**: Helps safely convert string representations of Python literals into actual Python objects
- **CountVectorizer**: Converts text documents into numerical feature vectors (Bag of Words)
- **cosine_similarity**: Calculates the cosine distance between vectors to measure similarity

---

## 2. Loading the Dataset

```python
movies = pd.read_csv('/kaggle/input/datasets/organizations/tmdb/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits = pd.read_csv('/kaggle/input/datasets/organizations/tmdb/tmdb-movie-metadata/tmdb_5000_credits.csv')
```

**Explanation:**
- Load two CSV files from the TMDB dataset:
  - **movies.csv**: Contains movie metadata (budget, genres, homepage, id, keywords, etc.)
  - **credits.csv**: Contains cast and crew information for each movie

---

## 3. Data Exploration & Merging

```python
movies.head(1)
credits.head(1)

# Merge datasets on 'title' column
movies = movies.merge(credits, on='title')
print(movies)
movies.info()
```

**Explanation:**
- `head(1)` displays the first row to understand the data structure
- `merge()` combines both DataFrames based on the common 'title' column
- Result: 4809 rows × 23 columns
- `info()` shows column names, non-null counts, and data types

---

## 4. Feature Selection

```python
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```

**Explanation:**
- Select only the relevant columns for our recommendation system:
  - **movie_id**: Unique identifier for each movie
  - **title**: Movie name (will be displayed to users)
  - **overview**: Brief plot summary
  - **genres**: Movie categories (Action, Comedy, etc.) - stored as JSON
  - **keywords**: Important terms associated with the movie - stored as JSON
  - **cast**: List of actors - stored as JSON
  - **crew**: Behind-the-scenes personnel (director, writers) - stored as JSON

---

## 5. Handling Missing Data

```python
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
```

**Explanation:**
- Check for null values (3 missing overviews out of 4809)
- Drop rows with missing data (insignificant loss)
- Verify no duplicate entries exist

---

## 6. Data Preprocessing (JSON Parsing)

The key challenge: genres, keywords, cast, and crew are stored as JSON strings that need to be parsed.

### Helper Function for JSON Parsing

```python
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):  # Safely converts string to Python object
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:             # Get only top 3 actors
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':    # Extract only the director
            L.append(i['name'])
            break
    return L
```

### Applying the Parsing Functions

```python
movies['keywords'] = movies['keywords'].apply(convert)
movies['genres'] = movies['genres'].apply(convert3)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
```

**Explanation:**
- **convert()**: Extracts all keyword names from JSON
- **convert3()**: Extracts only top 3 actors (reduces noise)
- **fetch_director()**: Extracts only the director's name from crew list

---

## 7. Feature Engineering

### Process Overview Text

```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
```

**Explanation:**
- Convert overview (string) into a list of words for easier processing

### Remove Spaces from Names

```python
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
```

**Explanation:**
- Remove spaces from multi-word names (e.g., "Sam Worthington" → "SamWorthington")
- This treats them as single entities/tags rather than separate words

### Create Tags Column

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string and lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

**Explanation:**
- Combine all features into a single 'tags' column
- Create a new DataFrame with only movie_id, title, and tags
- Convert the list of tags into a single space-separated string
- Convert all text to lowercase for consistency

---

## 8. Text Vectorization (Bag of Words)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer with 5000 most frequent words, removing English stopwords
cv = CountVectorizer(max_features=5000, stop_words='english')

# Transform tags into vectors
vectors = cv.fit_transform(new_df['tags']).toarray()

# View the vocabulary
cv.get_feature_names_out()
```

**Explanation:**
- **CountVectorizer** converts text into numerical vectors:
  - Builds a vocabulary of the 5000 most frequent words
  - Removes common English stop words (the, is, and, etc.)
  - Creates a matrix where each row represents a movie and each column represents a word
  - Values indicate how many times each word appears in the movie's tags

**Why 5000 words?**
- Reduces dimensionality while preserving important information
- Removes very rare words that might cause noise

---

## 9. Similarity Calculation

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```

**Explanation:**
- **Cosine similarity** measures the angle between two vectors
- Range: -1 (completely opposite) to 1 (identical)
- For text vectors, values typically range from 0 to 1
- Result: 4806×4806 matrix where similarity[i][j] = similarity between movie i and movie j
- Diagonal values are 1 (movie compared with itself)

---

## 10. Recommendation Function

```python
def recommend(movie):
    # Find the index of the movie
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Get similarity distances for this movie
    distances = similarity[movie_index]
    
    # Sort movies by similarity (excluding itself)
    movies_list = sorted(list(enumerate(distances)), 
                         reverse=True, 
                         key=lambda x: x[1])[1:6]
    
    # Print recommended movies
    print(f"Top 5 movies similar to '{movie}':")
    print("-" * 40)
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

**How it works:**
1. **Find movie index**: Locate the row index of the input movie
2. **Get distances**: Extract the similarity row for that movie
3. **Sort and filter**: 
   - `enumerate()` preserves original indices while sorting
   - Sort by similarity score in descending order
   - Exclude the first result (the movie itself)
   - Take the next 5 movies
4. **Display results**: Print the titles of recommended movies

---

## 11. Testing the Recommender

```python
recommend('Avatar')
```

**Example Output:**
```
Top 5 movies similar to 'Avatar':
----------------------------------------
Aliens vs Predator: Requiem
Independence Day
Battle: Los Angeles
Ender's Game
John Carter
```

---

## 🧠 How the System Works - Summary

1. **Data Collection**: Load TMDB movie dataset with metadata
2. **Feature Selection**: Choose relevant features (overview, genres, keywords, cast, crew)
3. **Data Cleaning**: Remove missing values and duplicates
4. **JSON Parsing**: Extract meaningful information from JSON-formatted columns
5. **Feature Combination**: Create a unified "tags" column containing all movie information
6. **Text Vectorization**: Convert tags into numerical vectors using Bag of Words
7. **Similarity Matrix**: Calculate cosine similarity between all movie pairs
8. **Recommendation**: Find movies with highest similarity scores to the input movie

---

## 📊 Key Concepts Explained

### Content-Based Filtering
- Recommends items similar to what the user has liked in the past
- Based on item features (not user behavior)
- Advantage: No need for other users' data
- Disadvantage: Limited to suggesting similar items (no serendipity)

### Bag of Words (BoW)
- Represents text as a multiset of words
- Ignores grammar and word order
- Keeps track of word frequencies
- Simple but effective for many text classification tasks

### Cosine Similarity
- Measures similarity between two non-zero vectors
- Formula: cos(θ) = (A·B) / (||A|| × ||B||)
- Insensitive to vector magnitude (focuses on direction)
- Ideal for text comparison where document length varies

### JSON Parsing with ast.literal_eval()
- Safely evaluates string containing Python literals
- More secure than eval() (doesn't execute arbitrary code)
- Converts JSON-like strings to actual Python objects

---

## 🚀 Possible Improvements

1. **Add more features**: Include release year, director, production companies
2. **Use TF-IDF instead of CountVectorizer**: Reduces importance of common words
3. **Implement collaborative filtering**: Combine with user behavior data
4. **Add sentiment analysis**: Process overview text for emotional tone
5. **Create a web interface**: Use Streamlit or Flask for user interaction
6. **Optimize performance**: Pre-compute and cache similarity matrix
7. **Add weighted features**: Give more importance to certain features (e.g., director)

---

## 📦 Dependencies

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- ast (built-in)

---

## 📝 Notes

- The similarity matrix is 4806×4806 (approximately 23 million values)
- Pre-computing this matrix allows for instant recommendations
- Memory usage: ~180MB for the similarity matrix
- The system can be easily deployed as a web application

---

**⭐ Star this notebook if you found it helpful!**
