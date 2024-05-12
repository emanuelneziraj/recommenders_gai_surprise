import os
import pandas as pd
from statistics import mean

def extract_movies_from_text(text):
    movies = []
    lines = text.split('\n')
    for line in lines:
        if '. ' in line:
            movie = line.split('. ', 1)[1].strip()
            movies.append(movie)
    return movies

def load_movie_recommendations(directory):
    user_movies = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            user_id = int(filename.split('_')[1])
            filepath = os.path.join(directory, filename)
            if os.stat(filepath).st_size == 0:
                continue
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            movies = extract_movies_from_text(text)
            user_movies[user_id] = movies
    return user_movies

def calculate_metrics_for_k(user_movies, actual_ratings, k):
    results = {}
    for user_id in sorted(user_movies.keys()):
        recommended_movies = user_movies[user_id][:k]
        actual_positive_movies = set(actual_ratings.get(user_id, []))
        recommended_movies_set = set(recommended_movies)
        true_positives = recommended_movies_set & actual_positive_movies
        precision = len(true_positives) / len(recommended_movies_set) if recommended_movies_set else 0
        recall = len(true_positives) / len(actual_positive_movies) if actual_positive_movies else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results[user_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    return results

def average_metrics(metrics):
    precision_list = [result['precision'] for result in metrics.values()]
    recall_list = [result['recall'] for result in metrics.values()]
    f1_list = [result['f1_score'] for result in metrics.values()]
    average_precision = mean(precision_list)
    average_recall = mean(recall_list)
    average_f1 = mean(f1_list)
    return average_precision, average_recall, average_f1

def evaluate(directory):
    directory = f'chat_gpt_output/{directory}/'
    ratings_file = '../ml-latest-small/ratings.csv'
    movies_file = '../ml-latest-small/movies.csv'

    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    ratings = ratings[ratings['rating'] >= 4.0].merge(movies, on='movieId')
    actual_ratings = ratings.groupby('userId')['title'].apply(list).to_dict()

    user_movies = load_movie_recommendations(directory)
    metrics_10 = calculate_metrics_for_k(user_movies, actual_ratings, 10)
    metrics_20 = calculate_metrics_for_k(user_movies, actual_ratings, 20)
    metrics_50 = calculate_metrics_for_k(user_movies, actual_ratings, 50)
    average_metrics_10 = average_metrics(metrics_10)
    average_metrics_20 = average_metrics(metrics_20)
    average_metrics_50 = average_metrics(metrics_50)

    output_file_path = "metrics_results.txt"
    with open(output_file_path, 'w') as file:
        for k, avg_metrics in zip([10, 20, 50], [average_metrics_10, average_metrics_20, average_metrics_50]):
            file.write(
                f"Average Metrics for GPT-3.5-Turbo @ {k}: {{'Precision': {avg_metrics[0]:.4f}, 'Recall': {avg_metrics[1]:.4f}, 'F1': {avg_metrics[2]:.4f}}}\n")
            print(
                f"Average Metrics for GPT-3.5-Turbo @ {k}: {{'Precision': {avg_metrics[0]:.4f}, 'Recall': {avg_metrics[1]:.4f}, 'F1': {avg_metrics[2]:.4f}}}")



evaluate("gpt-3.5-turbo")