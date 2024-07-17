import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity



books = pd.read_csv('notebook/data/Books.csv', low_memory=False)
ratings = pd.read_csv('notebook/data/Ratings.csv')
users = pd.read_csv('notebook/data/Users.csv')

books = books[['ISBN', 'Book-Title', 'Book-Author']]


with open('books.pkl', 'wb') as f:
    pickle.dump(books, f)

book_ratings = books.merge(ratings ,on = 'ISBN')
user_rating = users.merge(ratings , on = 'User-ID')

book_num_ratings = book_ratings.groupby(['Book-Title', 'Book-Author'])['Book-Rating'].count().reset_index().rename(columns\
    = {'Book-Rating':'Num-Ratings' })

book_avg_ratings = book_ratings.groupby(['Book-Title', 'Book-Author'])['Book-Rating'].mean().reset_index().rename(columns \
    = {'Book-Rating':'Avg-Ratings' })

final_rating = book_num_ratings.merge(book_avg_ratings , on = 'Book-Title')

popular_books = final_rating[final_rating['Num-Ratings'] > 250] \
    .sort_values(by = 'Avg-Ratings'  , ascending= False).reset_index(drop = True).head(50)
    
    
popular_books = popular_books.merge(books, on = "Book-Title")

# Save the DataFrame to a pickle file
with open('popular_books.pkl', 'wb') as f:
    pickle.dump(popular_books, f)
    
    
x = book_ratings.groupby('User-ID').count()['Book-Rating'] > 200
educated_users  = x[x].index

book_ratings = book_ratings[book_ratings['User-ID'].isin(educated_users)]

y  = book_ratings.groupby('Book-Title')['Book-Rating'].count() >= 50
famous_books = y[y].index

final = book_ratings[book_ratings['Book-Title'].isin(famous_books)]


pt = final.pivot_table(index = 'Book-Title' , columns = 'User-ID' , values= 'Book-Rating').fillna(0)


with open('pt.pkl', 'wb') as f:
    pickle.dump(pt, f)
    
similarity_scores = cosine_similarity(pt)

with open('similarity_scores.pkl', 'wb') as f:
    pickle.dump(similarity_scores, f)
    
    

    
#print(popular_books.head())