import streamlit as st
import pandas as pd

movies_list = pickle.load(open('movies.pkl', 'wb'))
movies_list = movies_list['title'].values

st.title("Movie Recommendation System")

option = st.selectbox()