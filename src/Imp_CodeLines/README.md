```python
movies.merge(credits, on='title')  # Both datasets merged to form 4809 rows x 23 cols

for i in ast.literal_eval(obj):  # Safely converts a string into a Python object

movies['overview'] = movies['overview'].apply(lambda x:x.split()) # To split the string into a list of words

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x]) # To remove spaces between names, so that it acts like a single tag(entity)

new_df['tags'].apply(lambda x:" ".join(x)) # convert list into a string

```
