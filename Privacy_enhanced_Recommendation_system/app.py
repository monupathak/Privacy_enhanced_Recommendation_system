import streamlit as st
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




import pandas as pd

with open('u.data', 'r') as file:
    # Read the lines of the file
    lines = file.readlines()

# Process the lines to create a list of tuples
data = []
for line in lines:
    parts = line.strip().split('\t')
    user_id = int(parts[0])
    item_id = int(parts[1])
    rating = int(parts[2])
    # Skip the timestamp (parts[3])
    data.append((user_id, item_id, rating))

# Create a DataFrame from the list of tuples
df = pd.DataFrame(data, columns=['userId', 'movieId', 'rating'])

# Print the first few rows of the DataFrame

df['movieId'].max()
df['movieId'].min()

temp = 1
userItem  = np.zeros((df['userId'].max()+1,df['movieId'].max()+1))
print(userItem )
print(userItem.shape)

for i in range (len(df['userId'])):
  userItem[df.userId[i]][df.movieId[i]] = df.rating[i]

print(userItem)
print(userItem.shape)


R = userItem[1:]
print(R)
R = R[:, 1:]
user_item_matrix=R


Rcap=np.loadtxt('/home/23m1526/100K/Rcap.csv', delimiter=',')
import pandas as pd
colName = ["movieId", "title", "Release Date", "Video Release Date",
           "IMDb URL", "Unknown", "Action", "Adventure", "Animation",
           "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
           "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
           "Thriller", "War", "Western"]

movies_df = pd.read_csv("u.item", sep="|", encoding="iso-8859-1",  names=colName)
movies_df.drop(columns= ["Release Date", "Video Release Date",
           "IMDb URL", "Unknown", "Action", "Adventure", "Animation",
           "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
           "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
           "Thriller", "War", "Western"], inplace=True)


# Sample user-item matrix (replace this with your actual data)
# user_item_matrix = pd.DataFrame({
#     'user_id': [1, 2, 3, 4],
#     'movieId': [101, 102, 103, 104],
#     'rating': [5, 4, 3, 5]
# })

# # Sample movies DataFrame (replace this with your actual data)
# movies_df = pd.DataFrame({
#     'movieId': [101, 102, 103, 104],
#     'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D']
# })

# Function to get recommendations for a given user ID
import numpy as np

def predRatings(user_item_matrix, Rcap, j, movies_df):
    def non_Rated_movie(user_item_matrix, j):
        zero_indices = np.where(user_item_matrix[:, j] == 0)[0]
        return zero_indices.tolist()

    def store_ratings_sorted(Rcap, zero_indices, j):
        ratings_dict = {}
        for i in zero_indices:
            rating = Rcap[i, j]
            ratings_dict[i] = rating
        sorted_ratings_dict = dict(sorted(ratings_dict.items(), key=lambda x: x[1], reverse=True))
        sorted_ratings_dict = dict(list(sorted_ratings_dict.items())[:20])
        return sorted_ratings_dict
#     st.write(movies_df.head())
    def get_movie_names(movies_df, top_20_ratings_dict):
        movie_names_dict = {}
        for key in top_20_ratings_dict.keys():
            movie_id = movies_df.loc[movies_df['movieId'] == key, 'title'].values[0]
            movie_names_dict[key] = movie_id
        return movie_names_dict

    zero_indices = non_Rated_movie(user_item_matrix, j)
    top_20_ratings_dict = store_ratings_sorted(Rcap, zero_indices, j)
    movie_names_dict = get_movie_names(movies_df, top_20_ratings_dict)

    return movie_names_dict

# p = predRatings(R,Rcap,8,movies_df)
# Streamlit UI
st.title('Movie Recommendation System')

# st.write(user_item_matrix['user_id'].max())
user_id = st.number_input('Enter User ID', min_value=1, max_value=600)
# user_id = st.number_input('Enter User ID', min_value=1, max_value=int(user_item_matrix['user_id'].max()))

if st.button('Get Recommendations'):
    recommendations = predRatings(R,Rcap,user_id,movies_df)
#     st.write(recommendations)
    if recommendations:
        st.subheader('Recommended Movies:')
#         st.table(recommendations[['movie_id', 'title']])
        st.write(recommendations)
    else:
        st.write('No recommendations found for the given user ID.')

