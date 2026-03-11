# Movie_Recommender_System

<br> ![Author: Deviprasad Shetty](https://img.shields.io/badge/Author-💫_Deviprasad%20Shetty-000000?style=for-the-badge&labelColor=white)
 
| [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://linkedin.com/in/deviprasad-shetty-4bba49313) | [![My_Portfolio](https://img.shields.io/badge/My_Portfolio-indigo?style=for-the-badge&logo=firefox&logoColor=white)](https://deviprasadshetty.com/) | [![My_Projects](https://img.shields.io/badge/My_Projects-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/DeviprasadShetty9833/My_Projects)  |                      
|---|---|---|

---

## 📌 Overview

A content-based Movie Recommender System analyzes movie metadata (genres, keywords, cast, crew) to find and suggest films similar to a user's selected movie. It uses **cosine similarity** on text data processed through **Count Vectorization** to compute movie similarities.

---

## ✨ Features

- 🔍 **Content-Based Filtering** – Recommends movies similar to a given title
- 🎭 **Multi-Factor Analysis** – Considers genres, keywords, cast, and crew
- 📊 **Similarity Scoring** – Ranks recommendations by relevance
- 🖥️ **Interactive Interface** – Simple web-based UI for easy interaction
- 📈 **Scalable Design** – Can handle large movie databases efficiently

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (CountVectorizer, cosine_similarity) |
| **Data Source** | TMDB 5000 Movie Dataset |
| **Version Control** | Git |

---

## 🧠 How It Works

### 1. **Data Collection**
   - Movie metadata from TMDB dataset
   - Includes: titles, overviews, genres, keywords, cast, crew

### 2. **Feature Engineering**
   - Combine relevant text features (genres + keywords + cast + crew)
   - Remove stopwords and unnecessary characters

### 3. **Vectorization**
   - Convert text features into numerical vectors using **CountVectorizer**
   - Each movie becomes a point in high-dimensional space

### 4. **Similarity Calculation**
   - Compute **cosine similarity** between all movie vectors
   - Higher cosine value = more similar movies

### 5. **Recommendation Engine**
   - User selects a movie
   - System finds top N most similar movies
   - Returns ranked recommendations

---

## 📁 Project Structure

```
movie-recommender-system/
│
├── app.py                 # Main Streamlit application
├── movie_data.csv         # Processed movie dataset
├── similarity.pkl         # Precomputed similarity matrix
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
│
├── notebooks/
│   └── data_preprocessing.ipynb  # EDA and feature engineering
│
└── assets/
    └── demo.gif           # Application demo
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**
   - Get [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) from Kaggle
   - Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project folder

4. **Run preprocessing (optional)**
   ```bash
   python preprocess.py
   ```

5. **Launch the app**
   ```bash
   streamlit run app.py
   ```

6. Open browser at `http://localhost:8501`

---

## 🎯 Usage Guide

1. **Select a movie** from the dropdown list
2. **Choose number of recommendations** (slider: 5-20)
3. **Click "Recommend"** to see suggestions
4. **View results** with movie posters and similarity scores

---

## 📊 Sample Output

```
Selected Movie: The Dark Knight

Top 5 Recommendations:
1. The Dark Knight Rises (Similarity: 0.85)
2. Batman Begins (Similarity: 0.82)
3. Inception (Similarity: 0.71)
4. The Prestige (Similarity: 0.68)
5. Iron Man (Similarity: 0.65)
```

---

## 📈 Performance

- **Dataset size**: 4800+ movies
- **Vector dimensions**: ~10,000 features
- **Response time**: <1 second (with precomputed similarity)
- **Memory usage**: ~200 MB for similarity matrix

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

