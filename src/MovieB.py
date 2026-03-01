import numpy as np
import pandas as pd
import ast           # To convert string to list
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)
credits.head(1)

movies = movies.merge(credits, on='title')  # Both datasets merged to form 4809 rows x 23 cols

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.isnull().sum()

movies.dropna(inplace=True)  # Since 3/5000 is missing, its insignificant, so we delete them

movies.duplicated().sum()    # No duplicates

movies.iloc[0].genres        # Finding loc by index

# We have to convert [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}] json format 
# to ["Action", "Adventure", "Fantasy", "Science Fiction"]

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):  # Safely converts a string into a Python object
        L.append(i['name'])
    return L

movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):  # Safely converts a string into a Python object
        if counter != 3:             # Bcoz we want the top 3 names of actors.
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['genres'] = movies['genres'].apply(convert3)
movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):  # Safely converts a string into a Python object
        if i['job'] == 'Director':   # Bcoz we want the top 3 names of actors.
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split()) # To split the string into a list of words

# To remove spaces between names, so that it acts like a single tag(entity)
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags' ]]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x)) # convert list into a string
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) # To make everything Lower Case

# Builds a vocabulary of 5000 words, Counts how often each word appears & Converts text into vectors. 
cv = CountVectorizer(max_features = 5000, stop_words = 'english') # stop_words Removes common useless words like: the, is, and, of, to, in
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors

cv.get_feature_names_out()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key = lambda x:x[1])[1:6] 

    for i in movies_list:
        
        print(new_df.iloc[i[0]].title)  # prints the original index of movies

recommend('Avatar')
