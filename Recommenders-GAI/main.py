from evaluate import evaluate
from open_ai_model import OpenAI
import pandas as pd
import os
import configparser

def send_message_to_chat_gpt(model, token):
    # Retrieve ratings and movies information
    ratings = pd.read_csv('../ml-latest-small/ratings.csv', sep=',',
                          header=0, names=['userId', 'movieId', 'rating', 'timestamp'],
                          usecols=['userId', 'movieId', 'rating'])

    movies = pd.read_csv('../ml-latest-small/movies.csv', sep=',', header=0,
                         names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])

    ratings = merge_titles_with_movies(ratings, movies)

    # Initialize variable for the request
    gAI = OpenAI(model, token)

    checkpoint_dir = f'chat_gpt_output/{model}/'

    for user in ratings['userId'].unique():
        print("Requesting recommendations for user {}".format(user))

        # Generate the message for ChatGPT
        message = movies_rated_by_user(ratings, user)
        print("Waiting for response...")

        # Send message to ChatGPT
        response = gAI.request(message)

        print("Response received")

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response)

def merge_titles_with_movies(ratings_df, movies_df):
    # Merge the two DataFrames on the 'MovieID' column
    ratings_df = pd.merge(ratings_df, movies_df, how='left', on='movieId')
    ratings_df = ratings_df.drop(columns='movieId')
    return ratings_df

def movies_rated_by_user(ratings, user):
    user_movies = ratings[ratings['userId'] == user]
    movie_ratings = user_movies[['title', 'rating']]
    movie_list = ', '.join(
        str(row['title']) + f" {int(row['rating'])}/5" for _, row in movie_ratings.iterrows())
    sentence = f"You know that the user {user} likes the following movies: {movie_list}."
    return sentence

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    token = config['DEFAULT']['GPT_TOKEN']
    model = "gpt-3.5-turbo" #gpt-3.5-turbo or gpt-4-turbo
    send_message_to_chat_gpt(model, token)
    evaluate(model)


if __name__ == '__main__':
    main()
    pass